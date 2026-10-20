#!/usr/bin/env python3
"""Federated free-rider detection (CSDD) on federated instruction tuning.

A small base LLM (default Qwen2.5-0.5B) is instruction-tuned with LoRA on
Dolly-15k by 20 clients with FedAvg.  One invocation runs one data partition
for a list of seeds and conditions, mirroring one partition of the
ECL-FL / 20NEWS-FL formal Shapley matrix:

    baseline (no attack) + {DFR, SDFR, AFR} x {10%, 30%, 50%} malicious

For a fixed (partition, seed), every condition shares the same client
partition, attacker order (10% subset of 30% subset of 50%), participation
schedule and initial LoRA state.  Every Shapley grouping ("axis") is scored
on the same client updates with the same permutations.  Each finished
condition+seed is written to ``<results-dir>/runs/<run>/`` and skipped on
restart; an unfinished one resumes from its last finished round.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from attacks.afr import (
    afr_attack,
    estimate_c_from_norms,
    estimate_decay_rate,
    estimate_e_cos_beta,
    global_update_norm,
    update_norm,
)
from attacks.dfr import dfr_attack, estimate_dfr_sigma, estimate_dfr_sigma_from_updates
from attacks.sdfr import sdfr_attack
from config import Config
from contribution.shapley import (
    EvalSet,
    Grouping,
    compute_class_metrics,
    estimate_round_shapley,
    example_losses,
    per_group_to_overall,
)
from data.dolly import CATEGORIES, load_dolly, load_reference_units, make_collate, random_labels_like
from fl.aggregation import fedavg_aggregate
from fl.client import FLClient
from fl.server import FLServer
from utils.partition import dirichlet_partition, iid_partition, partition_statistics
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("fedit-fl")

SCORE_COLUMNS = {
    # name: (column, lower_is_malicious)
    "V_mean": ("v_mean", True),          # paper Eq. 20, primary score
    "V_sum": ("v_sum", True),
    "P_mean": ("p_mean", True),
    "P_sum": ("p_sum", True),
    "SV_mean": ("sv_mean", True),
    "SV_sum": ("sv_sum", True),
}

AXIS_CHOICES = ("llm", "llm_random", "category", "random", "cluster", "global")


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# ── Shapley axes ───────────────────────────────────────────────────────────

def reference_labels(name: str, data, config) -> np.ndarray:
    """Group label of every reference row under a class axis."""
    if name == "llm":
        return load_reference_units(config.llm_units_file, data.splits["reference"])
    if name == "llm_random":
        return random_labels_like(reference_labels("llm", data, config), config.group_seed + 1)
    if name == "category":
        return data.reference.groups.copy()
    if name == "random":
        return random_labels_like(data.reference.groups, config.group_seed)
    if name == "cluster":
        present = np.unique(data.reference_clusters)
        return np.searchsorted(present, data.reference_clusters)
    raise ValueError(f"unknown class axis: {name!r}")


def build_groupings(axis_names, data, config) -> list[Grouping]:
    """Labels over the Shapley evaluation set = reference rows, then (when
    the global axis is used) the global rows."""
    num_reference = len(data.reference)
    num_global = len(data.global_set) if "global" in axis_names else 0
    groupings = []
    for name in axis_names:
        labels = np.full(num_reference + num_global, -1, dtype=np.int64)
        if name == "global":
            labels[num_reference:] = 0
            groupings.append(Grouping(name, labels, 1))
            continue
        labels[:num_reference] = reference_labels(name, data, config)
        groupings.append(Grouping(name, labels, int(labels.max()) + 1))
    return groupings


# ── naming (same conventions as 20NEWS-FL and ECL-FL) ──────────────────────

def alpha_tag(alpha: float) -> str:
    alpha = float(alpha)
    if alpha.is_integer():
        return f"{int(alpha)}p0"
    return f"{alpha:g}".replace(".", "p")


def partition_id(method: str, alpha: float | None) -> str:
    return "iid" if method == "iid" else f"dirichlet_alpha{alpha_tag(alpha)}"


def experiment_name(attack: str, ratio: float, part_id: str) -> str:
    if attack == "none":
        return f"baseline_no_attack_partition_{part_id}"
    return f"attack_{attack}_mal{int(round(ratio * 100))}_partition_{part_id}"


def build_participation_schedule(num_clients, ratio, num_rounds, seed):
    num_selected = max(1, int(round(num_clients * ratio)))
    rng = np.random.default_rng(seed)
    return [
        sorted(rng.choice(num_clients, num_selected, replace=False).tolist())
        for _ in range(num_rounds)
    ]


# ── attacks (logic copied from 20NEWS-FL ``_apply_attack``) ─────────────────

def apply_attack(attack_type, global_sd, global_history, config, round_index,
                 dfr_sigma_est, client_id, afr_c_est):
    effective = attack_type
    if attack_type in {"sdfr", "afr"} and round_index < config.attack_warmup_rounds:
        effective = "dfr"
    if effective == "dfr":
        sigma = config.dfr_sigma_override if config.dfr_sigma_override is not None else dfr_sigma_est
        if sigma is None:
            raise RuntimeError("DFR requires a bootstrap sigma")
        generator = torch.Generator().manual_seed(
            config.seed * 1_000_003 + round_index * 1_009 + client_id)
        return dfr_attack(global_sd, sigma=sigma, round_num=round_index + 1,
                          gamma=config.dfr_gamma, generator=generator), {"dfr_sigma": sigma}
    if effective == "sdfr":
        return sdfr_attack(global_sd, global_history), {}
    if attack_type == "afr":
        afr_c = config.afr_c_override if config.afr_c_override is not None else afr_c_est
        if afr_c is None:
            raise RuntimeError("AFR C was not estimated during warm-up")
        first_norm = global_update_norm(global_history[1], global_history[0])
        current_norm = global_update_norm(global_sd, global_history[-1])
        decay_rate = estimate_decay_rate(current_norm, first_norm, round_num=round_index)
        e_cos_beta = estimate_e_cos_beta(afr_c, decay_rate, round_num=round_index)
        update, base_norm = afr_attack(
            global_sd, global_history,
            n_total=config.num_clients,
            e_cos_beta=e_cos_beta,
            noisy_frac=config.afr_noisy_frac,
            seed=config.seed + round_index * config.num_clients + client_id,
        )
        return update, {"afr_base_norm": base_norm, "afr_decay_rate": decay_rate,
                        "afr_e_cos_beta": e_cos_beta}
    raise ValueError(f"unknown attack type: {attack_type}")


def assert_finite(update, label):
    for key, value in update.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} contains non-finite values in {key}")


# ── detection scores ─────────────────────────────────────────────────────

def pairwise_auc(benign, malicious, lower_is_malicious=True) -> float:
    """Pr(benign ranked safer than malicious); ties count one half."""
    b = np.asarray(benign, dtype=float)[:, None]
    m = np.asarray(malicious, dtype=float)[None, :]
    if b.size == 0 or m.size == 0:
        return float("nan")
    wins = (b > m) if lower_is_malicious else (b < m)
    return float((wins.sum() + 0.5 * (b == m).sum()) / (b.size * m.size))


def client_totals(round_details: list[dict], num_clients: int,
                  malicious_ids: set) -> list[dict]:
    frame = pd.DataFrame(round_details)
    rows = []
    for cid in range(num_clients):
        part = frame[frame["client_id"] == cid] if len(frame) else frame
        n = int(len(part))
        v_sum = float(part["class_sv_variance"].sum()) if n else 0.0
        p_sum = float(part["positive_class_sv_sum"].sum()) if n else 0.0
        sv_sum = float(part["round_shapley_value"].sum()) if n else 0.0
        rows.append({
            "client_id": cid,
            "is_malicious": cid in malicious_ids,
            "participations": n,
            "v_sum": v_sum,
            "v_mean": v_sum / n if n else 0.0,
            "p_sum": p_sum,
            "p_mean": p_sum / n if n else 0.0,
            "sv_sum": sv_sum,
            "sv_mean": sv_sum / n if n else 0.0,
        })
    return rows


def auc_record(totals: list[dict], suffix: str = "") -> dict:
    frame = pd.DataFrame(totals)
    malicious = frame["is_malicious"].astype(bool)
    record = {}
    for name, (column, lower) in SCORE_COLUMNS.items():
        record[f"auc_{name}{suffix}"] = pairwise_auc(
            frame.loc[~malicious, column], frame.loc[malicious, column], lower
        )
    return record


def axis_client_totals(axis_records: list[dict], axis_names, num_clients: int,
                       malicious_ids: set) -> list[dict]:
    """``client_totals`` for every Shapley axis, with an ``axis`` column."""
    rows = []
    for name in axis_names:
        part = [r for r in axis_records if r["axis"] == name]
        rows += [dict(row, axis=name)
                 for row in client_totals(part, num_clients, malicious_ids)]
    return rows


# ── round-level checkpoint ──────────────────────────────────────────────────

CHECKPOINT_VERSION = 1


def checkpoint_identity(config) -> dict:
    """Fields that must match for a checkpoint to be resumed."""
    keys = ("experiment_name", "seed", "attack_type", "malicious_ratio", "num_rounds",
            "num_clients", "participation_ratio", "partition_id", "num_mc_samples",
            "local_steps", "batch_size", "micro_batch_size", "local_lr", "local_lr_schedule",
            "model_path", "lora_r", "lora_alpha", "lora_target_modules", "max_length",
            "max_response_length", "reference_per_category", "global_size", "global_holdout",
            "test_size",
            "split_seed", "partition_clusters", "shapley_axes", "llm_units_sha256",
            "compute_shapley")
    identity = {key: getattr(config, key) for key in keys}
    identity["lora_target_modules"] = list(identity["lora_target_modules"])
    identity["shapley_axes"] = list(identity["shapley_axes"])
    return identity


def save_checkpoint(path: str, payload: dict) -> None:
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str, config):
    if not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # truncated or corrupt file: start the run over
        logger.warning("ignoring unreadable checkpoint %s (%s)", path, exc)
        return None
    if (payload.get("version") != CHECKPOINT_VERSION
            or payload.get("identity") != checkpoint_identity(config)):
        logger.warning("ignoring checkpoint %s: configuration changed", path)
        return None
    return payload


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ── one run ───────────────────────────────────────────────────────────────

class RunContext:
    """Everything that is shared by the runs of one process."""

    def __init__(self, model, initial_state, data, collate, shapley_eval, reference_eval,
                 test_eval, groupings, client_grouping):
        self.model = model
        self.initial_state = initial_state
        self.data = data
        self.collate = collate
        self.shapley_eval = shapley_eval        # reference rows (+ global rows)
        self.reference_eval = reference_eval    # reference rows only
        self.test_eval = test_eval
        self.groupings = groupings              # primary first
        # Grouping of the reference rows for the per-client and per-round
        # reference metrics (the primary axis unless that is "global").
        self.client_grouping = client_grouping


def run_experiment(config, ctx: RunContext, partition, malicious_ids, schedule,
                   checkpoint_path=None):
    """Run one condition+seed; resumable after every finished round."""
    groupings = ctx.groupings
    primary = groupings[0]
    model = ctx.model
    set_seed(config.seed)
    server = FLServer(model, ctx.test_eval, ctx.data.test.groups, len(CATEGORIES), config)
    server.global_state_dict = OrderedDict((k, v.clone()) for k, v in ctx.initial_state.items())
    clients = {
        cid: FLClient(cid, Subset(ctx.data.train, indices), ctx.collate, config)
        for cid, indices in partition.items()
    }
    sample_counts = {cid: len(indices) for cid, indices in partition.items()}
    client_grouping = ctx.client_grouping
    num_mal = len(malicious_ids)
    common = {
        "experiment_name": config.experiment_name,
        "attack_type": config.attack_type,
        "run_seed": config.seed,
        "malicious_ratio": config.malicious_ratio if num_mal else 0.0,
        "partition_id": config.partition_id,
        "dirichlet_alpha": config.dirichlet_alpha,
    }

    state = {
        "dfr_sigma_est": None,
        "afr_c_est": None,
        "last_warmup_mean_local_norm": None,
        "participation": {cid: 0 for cid in range(config.num_clients)},
        "cumulative_sv": {cid: 0.0 for cid in range(config.num_clients)},
        "class_var_history": {cid: [] for cid in range(config.num_clients)},
        "pos_sum_history": {cid: [] for cid in range(config.num_clients)},
        "records": {key: [] for key in RECORD_KEYS},
        "elapsed_seconds": 0.0,
    }
    start_round = 0
    resumed = load_checkpoint(checkpoint_path, config) if checkpoint_path else None
    if resumed is not None:
        start_round = resumed["next_round"]
        state = resumed["state"]
        server.global_state_dict = resumed["global_state"]
        server.global_history.clear()
        server.global_history.extend(resumed["global_history"])
        for cid, generator_state in resumed["client_generators"].items():
            clients[cid].data_loader.generator.set_state(generator_state)
        restore_rng_state(resumed["rng"])
        logger.info("resume %s seed=%d from round %d", config.experiment_name,
                    config.seed, start_round)
    else:
        # The untrained model (LoRA B = 0 = the base model): the reference
        # point of round0_share_of_loss_drop.
        initial_loss, initial_groups = server.evaluate_detailed()
        state["initial_test_loss"] = initial_loss
        state["initial_test_group_loss"] = initial_groups.tolist()
        logger.info("%s seed=%d untrained test_loss=%.4f", config.experiment_name,
                    config.seed, initial_loss)

    records = state["records"]
    participation = state["participation"]
    for round_t in range(start_round, config.num_rounds):
        try:
            started = time.perf_counter()
            selected = schedule[round_t]
            global_sd = server.get_global_state_dict()
            global_history = list(server.global_history)

            if (config.attack_type == "afr" and state["afr_c_est"] is None
                    and config.afr_c_override is None
                    and round_t >= config.attack_warmup_rounds):
                first_norm = global_update_norm(global_history[1], global_history[0])
                current_norm = global_update_norm(global_sd, global_history[-1])
                decay_rate = estimate_decay_rate(current_norm, first_norm, round_num=round_t)
                state["afr_c_est"] = estimate_c_from_norms(
                    state["last_warmup_mean_local_norm"], current_norm,
                    config.num_clients, decay_rate, round_num=round_t - 1,
                )
            if (config.attack_type in {"dfr", "afr"} and state["dfr_sigma_est"] is None
                    and config.dfr_sigma_override is None):
                state["dfr_sigma_est"] = estimate_dfr_sigma(global_sd, global_history)

            updates = {}
            train_loss = {}
            attacking = set(malicious_ids).intersection(selected)
            honest_ids = []
            for cid in selected:
                if cid in malicious_ids:
                    continue
                participation[cid] += 1
                updates[cid] = clients[cid].train(model, global_sd)
                train_loss[cid] = clients[cid].last_train_loss
                assert_finite(updates[cid], f"client {cid} update")
                honest_ids.append(cid)
            if (config.attack_type in {"dfr", "sdfr", "afr"} and state["dfr_sigma_est"] is None
                    and config.dfr_sigma_override is None):
                state["dfr_sigma_est"] = estimate_dfr_sigma_from_updates(
                    {cid: updates[cid] for cid in honest_ids}
                )
                if state["dfr_sigma_est"] is None or state["dfr_sigma_est"] <= 0.0:
                    raise RuntimeError("could not bootstrap a positive DFR sigma")
            for cid in selected:
                if cid not in malicious_ids:
                    continue
                participation[cid] += 1
                updates[cid], _ = apply_attack(
                    config.attack_type, global_sd, global_history, config,
                    round_t, state["dfr_sigma_est"], cid, state["afr_c_est"],
                )
                train_loss[cid] = float("nan")   # free riders do not train
                assert_finite(updates[cid], f"client {cid} update")
            if config.attack_type == "afr" and round_t < config.attack_warmup_rounds:
                if not honest_ids:
                    raise RuntimeError("AFR warm-up requires an honest selected client")
                state["last_warmup_mean_local_norm"] = float(
                    np.mean([update_norm(updates[cid]) for cid in honest_ids])
                )
            client_seconds = time.perf_counter() - started

            shapley_started = time.perf_counter()
            per_axis = {}
            evaluations = 0
            if config.compute_shapley:
                axis_sv, evaluations = estimate_round_shapley(
                    model, updates, global_sd, ctx.shapley_eval, groupings,
                    num_mc_samples=config.num_mc_samples,
                    device=config.device,
                    sample_counts=None if config.shapley_equal_weights else {
                        cid: sample_counts[cid] for cid in selected},
                    cache_coalitions=config.shapley_cache_coalitions,
                    rng=np.random.default_rng(
                        (int(config.seed) + 7) * 1_000_003 + (round_t + 1) * 10_007),
                )
                for grouping in groupings:
                    values = axis_sv[grouping.name]
                    per_axis[grouping.name] = (values,
                                               per_group_to_overall(values, grouping.weights),
                                               compute_class_metrics(values))
                per_group, overall, metrics = per_axis[primary.name]
            else:
                per_group, overall, metrics = {}, {}, {}
            shapley_seconds = time.perf_counter() - shapley_started

            # Per-group reference loss of every submitted local model (global +
            # update): the counterpart of per-class accuracy/loss per client.
            client_eval_started = time.perf_counter()
            client_losses = {}
            for cid in selected:
                local_state = OrderedDict(
                    (key, value + updates[cid][key]) for key, value in global_sd.items())
                client_losses[cid] = client_grouping.means(
                    example_losses(model, local_state, ctx.reference_eval, config.device))
            client_eval_seconds = time.perf_counter() - client_eval_started

            aggregation_started = time.perf_counter()
            new_sd = fedavg_aggregate(
                global_sd, updates, weights={cid: sample_counts[cid] for cid in selected})
            assert_finite(new_sd, "aggregated global state")
            server.update_global_model(new_sd)
            test_loss, test_group_loss = server.evaluate_detailed()
            ref_group_loss = client_grouping.means(
                example_losses(model, server.global_state_dict, ctx.reference_eval,
                               config.device))
            aggregation_eval_seconds = time.perf_counter() - aggregation_started
            round_seconds = time.perf_counter() - started
            logger.info(
                "%s seed=%d round %2d: test_loss=%.4f clients=%.1fs shapley=%.1fs "
                "(%d coalitions) total=%.1fs",
                config.experiment_name, config.seed, round_t, test_loss,
                client_seconds, shapley_seconds, evaluations, round_seconds,
            )

            curve = dict(common, round=round_t, test_loss=test_loss,
                         test_ppl=float(math.exp(min(test_loss, 50.0))))
            curve.update({f"test_group_{g}_loss": float(v) for g, v in enumerate(test_group_loss)})
            curve.update({f"ref_group_{g}_loss": float(v) for g, v in enumerate(ref_group_loss)})
            curve.update({"client_phase_seconds": client_seconds,
                          "shapley_phase_seconds": shapley_seconds,
                          "shapley_evaluations": evaluations,
                          "client_eval_seconds": client_eval_seconds,
                          "aggregation_eval_seconds": aggregation_eval_seconds,
                          "round_seconds": round_seconds})
            records["round_curves"].append(curve)

            for cid in selected:
                m = metrics.get(cid, {})
                sv = overall.get(cid, 0.0)
                pc = participation[cid]
                state["cumulative_sv"][cid] += sv
                state["class_var_history"][cid].append(m.get("class_sv_variance", 0.0))
                state["pos_sum_history"][cid].append(m.get("positive_class_sv_sum", 0.0))
                group_loss = client_losses[cid]
                records["round_details"].append(dict(
                    common,
                    round=round_t,
                    client_id=cid,
                    test_loss=test_loss,
                    is_malicious=cid in malicious_ids,
                    is_attacking=cid in attacking,
                    participation_count_so_far=pc,
                    num_samples=sample_counts[cid],
                    round_shapley_value=sv,
                    cumulative_shapley_value=state["cumulative_sv"][cid],
                    mean_shapley_value=state["cumulative_sv"][cid] / pc,
                    class_sv_variance=m.get("class_sv_variance", 0.0),
                    positive_class_sv_sum=m.get("positive_class_sv_sum", 0.0),
                    concentration_ratio=m.get("concentration_ratio", 0.0),
                    dominant_class=m.get("dominant_class", -1),
                    mean_class_sv_variance=float(np.mean(state["class_var_history"][cid])),
                    mean_positive_class_sv_sum=float(np.mean(state["pos_sum_history"][cid])),
                    local_train_loss=train_loss[cid],
                    local_ref_loss=float(np.dot(client_grouping.weights, group_loss)),
                    update_norm=float(update_norm(updates[cid])),
                    client_phase_seconds=client_seconds,
                    shapley_phase_seconds=shapley_seconds,
                    aggregation_eval_seconds=aggregation_eval_seconds,
                    round_seconds=round_seconds,
                ))
                if cid in per_group:
                    record = dict(common, round=round_t, client_id=cid,
                                  is_malicious=cid in malicious_ids)
                    record.update({f"class_{g}": float(v) for g, v in enumerate(per_group[cid])})
                    records["per_class_records"].append(record)
                record = dict(common, round=round_t, client_id=cid,
                              is_malicious=cid in malicious_ids,
                              is_attacking=cid in attacking,
                              local_train_loss=train_loss[cid])
                record.update({f"class_{g}_loss": float(v) for g, v in enumerate(group_loss)})
                records["per_class_client_metrics"].append(record)
                for grouping in groupings:
                    if cid not in per_axis.get(grouping.name, ({},))[0]:
                        continue
                    values, axis_overall, axis_metrics = per_axis[grouping.name]
                    record = dict(common, round=round_t, client_id=cid,
                                  is_malicious=cid in malicious_ids,
                                  is_attacking=cid in attacking,
                                  axis=grouping.name, num_groups=grouping.num_groups,
                                  round_shapley_value=axis_overall[cid])
                    record.update(axis_metrics[cid])
                    record.update({f"class_{g}": float(v) for g, v in enumerate(values[cid])})
                    records["axis_records"].append(record)

            state["elapsed_seconds"] += round_seconds
            if checkpoint_path:
                save_checkpoint(checkpoint_path, {
                    "version": CHECKPOINT_VERSION,
                    "identity": checkpoint_identity(config),
                    "next_round": round_t + 1,
                    "state": state,
                    "global_state": server.global_state_dict,
                    "global_history": list(server.global_history),
                    "client_generators": {cid: c.data_loader.generator.get_state()
                                          for cid, c in clients.items()},
                    "rng": rng_state(),
                })
        except FloatingPointError as exc:
            # The global model diverged (e.g. under a strong attack): keep the
            # rounds completed so far, record where it happened and stop.
            state["diverged_round"] = round_t
            state["diverged_reason"] = str(exc)
            logger.warning("%s seed=%d diverged at round %d: %s", config.experiment_name,
                           config.seed, round_t, exc)
            break

    return state


RECORD_KEYS = ("round_details", "per_class_records", "per_class_client_metrics",
               "round_curves", "axis_records")


# ── per-run outputs (written as soon as a condition+seed finishes) ─────────

def round0_scale(details: pd.DataFrame) -> float:
    """Mean V of honest clients in round 0 over their mean V from round 3 on
    (how much round 0 dominates the raw participation means)."""
    if not len(details) or "class_sv_variance" not in details:
        return float("nan")
    honest = details[~details["is_malicious"].astype(bool)]
    early = honest.loc[honest["round"] == 0, "class_sv_variance"].mean()
    late = honest.loc[honest["round"] >= 3, "class_sv_variance"].mean()
    return float(early / late) if late and np.isfinite(late) and late > 0 else float("nan")


def experiment_summary(config, state, malicious_ids, totals, num_axis_groups,
                       wall_seconds) -> dict:
    details = pd.DataFrame(state["records"]["round_details"])
    curves = pd.DataFrame(state["records"]["round_curves"])
    frame = pd.DataFrame(totals)
    honest = frame[~frame["is_malicious"]]
    malicious = frame[frame["is_malicious"]]
    seen_h = honest[honest["participations"] > 0]
    seen_m = malicious[malicious["participations"] > 0]

    def mean(part, column):
        return float(part[column].mean()) if len(part) else 0.0

    nan = float("nan")
    if not len(curves):   # diverged in its first round: nothing to summarize
        curves = pd.DataFrame({c: [nan] for c in (
            "test_loss", "test_ppl", "round_seconds", "client_phase_seconds",
            "shapley_phase_seconds", "shapley_evaluations", "client_eval_seconds",
            "aggregation_eval_seconds")})
        details = pd.DataFrame({"is_malicious": pd.Series(dtype=bool),
                                "local_ref_loss": pd.Series(dtype=float),
                                "round": pd.Series(dtype=int)})

    initial = state.get("initial_test_loss", nan)
    after_round0 = float(curves["test_loss"].iloc[0])
    final = float(curves["test_loss"].iloc[-1])
    total_drop = initial - final
    num_mal = len(malicious_ids)
    summary = {
        "experiment_name": config.experiment_name,
        "attack_type": config.attack_type,
        "run_seed": config.seed,
        "partition_seed": config.partition_seed,
        "partition_id": config.partition_id,
        "malicious_ratio": config.malicious_ratio if num_mal else 0.0,
        "num_malicious_clients": num_mal,
        "partition_method": config.partition_method,
        "dirichlet_alpha": config.dirichlet_alpha,
        "malicious_client_ids": ",".join(map(str, sorted(malicious_ids))),
        "shapley_axis": config.shapley_axis,
        "shapley_axes": ",".join(config.shapley_axes),
        "num_axis_groups": num_axis_groups,
        "num_rounds": config.num_rounds,
        "num_clients": config.num_clients,
        "participation_ratio": config.participation_ratio,
        "local_lr": config.local_lr,
        "untrained_test_loss": initial,
        "initial_test_loss": after_round0,
        "final_test_loss": final,
        "final_test_ppl": float(math.exp(min(final, 50.0))) if np.isfinite(final) else nan,
        "best_test_loss": float(curves["test_loss"].min()),
        "round0_share_of_loss_drop": (float((initial - after_round0) / total_drop)
                                      if np.isfinite(total_drop) and total_drop > 0 else nan),
        "round0_v_scale": round0_scale(details),
        "diverged": "diverged_round" in state,
        "diverged_round": state.get("diverged_round"),
        "diverged_reason": state.get("diverged_reason"),
        "rounds_completed": int(curves["test_loss"].notna().sum()),
        "avg_round_shapley_honest": mean(seen_h, "sv_mean"),
        "avg_round_shapley_malicious": mean(seen_m, "sv_mean"),
        "avg_cumulative_shapley_honest": mean(honest, "sv_sum"),
        "avg_cumulative_shapley_malicious": mean(malicious, "sv_sum"),
        "shapley_gap_honest_vs_malicious": mean(seen_h, "sv_mean") - mean(seen_m, "sv_mean"),
        "avg_class_sv_variance_honest": mean(seen_h, "v_mean"),
        "avg_class_sv_variance_malicious": mean(seen_m, "v_mean"),
        "avg_positive_class_sv_sum_honest": mean(seen_h, "p_mean"),
        "avg_positive_class_sv_sum_malicious": mean(seen_m, "p_mean"),
        "avg_local_ref_loss_honest": float(details.loc[~details["is_malicious"].astype(bool),
                                                       "local_ref_loss"].mean()),
        "avg_local_ref_loss_malicious": (float(details.loc[details["is_malicious"].astype(bool),
                                                           "local_ref_loss"].mean())
                                         if num_mal else nan),
        "dfr_sigma": state["dfr_sigma_est"],
        "afr_c": state["afr_c_est"],
        "total_experiment_seconds": float(curves["round_seconds"].sum()),
        "wall_seconds_this_process": wall_seconds,
        "mean_round_seconds": float(curves["round_seconds"].mean()),
        "mean_client_phase_seconds": float(curves["client_phase_seconds"].mean()),
        "mean_shapley_phase_seconds": float(curves["shapley_phase_seconds"].mean()),
        "mean_shapley_evaluations": float(curves["shapley_evaluations"].mean()),
        "mean_client_eval_seconds": float(curves["client_eval_seconds"].mean()),
        "mean_aggregation_eval_seconds": float(curves["aggregation_eval_seconds"].mean()),
    }
    if malicious_ids and config.compute_shapley:
        summary.update(auc_record(totals))
        axis_totals = pd.DataFrame(state.get("axis_totals", []))
        for name, part in (axis_totals.groupby("axis", sort=False) if len(axis_totals) else []):
            record = auc_record(part.to_dict("records"), suffix=f"__{name}")
            if name == "global":   # one group: V and P are undefined
                record = {k: v for k, v in record.items() if k.startswith("auc_SV")}
            summary.update(record)
    return summary


def finish_run(config, run_dir, state, malicious_ids, num_axis_groups,
               partition_frame, config_record, wall_seconds, make_plots=True):
    """Write CSVs, the 20NEWS-style workbook and plots; mark the run done."""
    from utils.export import write_run_workbook

    records = state["records"]
    totals = client_totals(records["round_details"], config.num_clients, malicious_ids)
    common = {"experiment_name": config.experiment_name,
              "partition_id": config.partition_id,
              "attack_type": config.attack_type,
              "malicious_ratio": config.malicious_ratio if malicious_ids else 0.0,
              "run_seed": config.seed}
    state["axis_totals"] = axis_client_totals(
        records["axis_records"], config.shapley_axes, config.num_clients,
        malicious_ids) if records["axis_records"] else []
    tables = {key: pd.DataFrame(records[key]) for key in RECORD_KEYS}
    tables["client_totals"] = pd.DataFrame(totals).assign(**common)
    tables["axis_client_totals"] = pd.DataFrame(state["axis_totals"]).assign(**common)
    for key, frame in tables.items():
        frame.to_csv(os.path.join(run_dir, f"{key}.csv"), index=False)
    summary = experiment_summary(config, state, malicious_ids, totals,
                                 num_axis_groups, wall_seconds)
    run_config = dict(config_record)
    run_config.update(asdict(config))
    write_run_workbook(os.path.join(run_dir, "experiment_results.xlsx"), tables,
                       summary, partition_frame, run_config)
    if make_plots:
        try:
            from visualization.plots import plot_run
            plot_run(run_dir, tables, summary)
        except Exception:  # plots must never cost a finished run
            logger.exception("plotting failed for %s", run_dir)
    write_json(os.path.join(run_dir, "summary.json"), summary)
    checkpoint = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    logger.info("done %s seed=%d final_loss=%.4f %s", config.experiment_name,
                config.seed, summary["final_test_loss"],
                {k: round(v, 3) for k, v in summary.items()
                 if k.startswith("auc_V_mean") or k in ("auc_P_mean", "auc_SV_mean")})


def refresh_summary(root: str, make_plots: bool) -> None:
    """Rebuild the combined workbook (and plots) after every finished run."""
    try:
        import summarize
        summarize.main(["--results-root", root] + ([] if make_plots else ["--no-plots"]))
    except SystemExit:
        pass
    except Exception:
        logger.exception("could not refresh the combined summary under %s", root)


# ── CLI ───────────────────────────────────────────────────────────────────

def csv_list(cast):
    return lambda text: tuple(cast(item) for item in text.split(",") if item.strip())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=Config.data_path)
    p.add_argument("--model-path", default=Config.model_path)
    p.add_argument("--results-dir", default="results/run")
    p.add_argument("--partition-method", choices=("iid", "dirichlet"), default="iid")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--attack-types", type=csv_list(str), default=("dfr", "sdfr", "afr"))
    p.add_argument("--malicious-ratios", type=csv_list(float), default=(0.1, 0.3, 0.5))
    p.add_argument("--experiment-seeds", type=csv_list(int), default=(42, 43, 44))
    p.add_argument("--no-baseline", action="store_true", help="skip the no-attack condition")
    p.add_argument("--shapley-axes", type=csv_list(str), default=Config.shapley_axes,
                   help="comma list of groupings scored on the same runs, primary first "
                        f"(choices: {','.join(AXIS_CHOICES)})")
    p.add_argument("--llm-units-file", default=Config.llm_units_file,
                   help="frozen reference-row-to-unit CSV for the llm axes")
    p.add_argument("--no-shapley", action="store_true")
    p.add_argument("--num-rounds", type=int, default=Config.num_rounds)
    p.add_argument("--num-clients", type=int, default=Config.num_clients)
    p.add_argument("--participation-ratio", type=float, default=Config.participation_ratio)
    p.add_argument("--num-mc-samples", type=int, default=Config.num_mc_samples)
    p.add_argument("--local-steps", type=int, default=Config.local_steps)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--micro-batch-size", type=int, default=Config.micro_batch_size)
    p.add_argument("--local-lr", type=float, default=Config.local_lr)
    p.add_argument("--lora-r", type=int, default=Config.lora_r)
    p.add_argument("--lora-alpha", type=float, default=Config.lora_alpha)
    p.add_argument("--max-length", type=int, default=Config.max_length)
    p.add_argument("--max-response-length", type=int, default=Config.max_response_length)
    p.add_argument("--reference-per-category", type=int, default=Config.reference_per_category)
    p.add_argument("--global-size", type=int, default=Config.global_size,
                   help="rows of the global axis (natural proportions); 0 drops the axis")
    p.add_argument("--test-size", type=int, default=Config.test_size)
    p.add_argument("--min-samples-per-client", type=int, default=Config.min_samples_per_client)
    p.add_argument("--eval-token-budget", type=int, default=Config.eval_token_budget)
    p.add_argument("--no-sha-check", action="store_true",
                   help="accept a data file other than the original Dolly release (tests)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--summary-root", default=None,
                   help="directory whose combined workbook is refreshed after "
                        "every finished run (default: --results-dir)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def resolve_device(choice: str, require_cuda: bool) -> str:
    if require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require-cuda was given but CUDA is not available")
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_json(path, payload):
    with open(path + ".tmp", "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    os.replace(path + ".tmp", path)


def check_same_configuration(config) -> None:
    """Refuse a results folder whose finished runs used other settings.

    Finished runs are skipped by name, so a changed learning rate, axis list,
    global size, model, ... must not reuse a folder (start.sh and the run
    scripts add a suffix such as ``_lr2e-4`` for that).
    """
    path = os.path.join(config.results_dir, "experiment_config.json")
    if not os.path.exists(path):
        return
    with open(path) as handle:
        previous = json.load(handle)
    run_specific = {"experiment_name", "seed", "attack_type", "malicious_ratio"}
    changed = {}
    for key, value in checkpoint_identity(config).items():
        if key in run_specific or key not in previous:
            continue
        if json.loads(json.dumps(previous[key], default=float)) != \
                json.loads(json.dumps(value, default=float)):
            changed[key] = (previous[key], value)
    if changed:
        raise SystemExit(f"{config.results_dir} holds runs with other settings {changed}; "
                         "use another --results-dir (or remove that folder)")


def partition_frame(data, partition) -> pd.DataFrame:
    """Per-client counts of partition clusters and of native categories."""
    clusters = pd.DataFrame(partition_statistics(data.partition_clusters, partition,
                                                 len(data.cluster_sizes)))
    clusters = clusters.rename(columns=lambda c: c.replace("class_", "cluster_"))
    categories = pd.DataFrame(partition_statistics(data.train.groups, partition, len(CATEGORIES)))
    categories = categories.drop(columns=["total_samples", "num_present_classes"]).rename(
        columns=lambda c: c.replace("class_", "category_"))
    return clusters.merge(categories, on="client_id")


def main(argv=None):
    args = parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device, args.require_cuda)
    axis_names = tuple(args.shapley_axes)
    if args.global_size <= 0 and "global" in axis_names:
        axis_names = tuple(a for a in axis_names if a != "global")
        logger.info("--global-size 0: the global axis is left out")
    unknown = [name for name in axis_names if name not in AXIS_CHOICES]
    if not axis_names or unknown or len(set(axis_names)) != len(axis_names):
        raise SystemExit(f"--shapley-axes must be distinct names from {AXIS_CHOICES}")
    part_id = partition_id(args.partition_method, args.dirichlet_alpha)
    uses_llm = bool({"llm", "llm_random"} & set(axis_names))

    base = Config(
        data_path=args.data_path,
        check_data_sha256=not args.no_sha_check,
        model_path=args.model_path,
        results_dir=args.results_dir,
        num_clients=args.num_clients,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha if args.partition_method == "dirichlet" else None,
        partition_id=part_id,
        num_rounds=args.num_rounds,
        participation_ratio=args.participation_ratio,
        num_mc_samples=args.num_mc_samples,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        local_lr=args.local_lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        max_response_length=args.max_response_length,
        reference_per_category=args.reference_per_category,
        global_size=args.global_size if "global" in axis_names else 0,
        test_size=args.test_size,
        min_samples_per_client=args.min_samples_per_client,
        eval_token_budget=args.eval_token_budget,
        shapley_axes=axis_names,
        llm_units_file=args.llm_units_file,
        llm_units_sha256=file_sha256(args.llm_units_file) if uses_llm else "",
        compute_shapley=not args.no_shapley,
        device=device,
    )
    os.makedirs(os.path.join(base.results_dir, "runs"), exist_ok=True)
    logger.info("device=%s partition=%s results=%s", device, part_id, base.results_dir)

    from models.llm import create_model, load_tokenizer

    tokenizer = load_tokenizer(base.model_path)
    data = load_dolly(base, tokenizer)
    collate = make_collate(tokenizer.pad_token_id)
    logger.info("Dolly: train=%d reference=%d global=%d test=%d; partition clusters %s",
                len(data.train), len(data.reference),
                len(data.global_set) if data.global_set is not None else 0,
                len(data.test), data.cluster_sizes)

    groupings = build_groupings(axis_names, data, base)
    for grouping in groupings:
        logger.info("Shapley axis %s: %d groups, rows per group %s", grouping.name,
                    grouping.num_groups, grouping.counts.tolist())
    reference_items = [data.reference[i] for i in range(len(data.reference))]
    shapley_items = list(reference_items)
    if "global" in axis_names:
        shapley_items += [data.global_set[i] for i in range(len(data.global_set))]
    shapley_eval = EvalSet(shapley_items, collate, device, base.eval_token_budget)
    reference_eval = EvalSet(reference_items, collate, device, base.eval_token_budget)
    test_eval = EvalSet([data.test[i] for i in range(len(data.test))], collate, device,
                        base.eval_token_budget)
    client_axis = groupings[0] if groupings[0].name != "global" else \
        build_groupings(("category",), data, base)[0]
    client_grouping = Grouping(client_axis.name, client_axis.labels[:len(data.reference)],
                               client_axis.num_groups)

    reference_table = data.frame.set_index("row_id").loc[data.splits["reference"],
                                                         ["category"]].reset_index()
    for grouping in groupings:
        if grouping.name != "global":
            reference_table[grouping.name] = grouping.labels[:len(data.reference)]
    reference_table.to_csv(os.path.join(base.results_dir, "reference_groups.csv"), index=False)
    pd.DataFrame({split: pd.Series(rows) for split, rows in data.splits.items()}).to_csv(
        os.path.join(base.results_dir, "splits.csv"), index=False)

    set_seed(0)
    model = create_model(base, device)
    initial_state = model.get_state_dict()

    attacks = [("none", 0.0)] if not args.no_baseline else []
    attacks += [(a, r) for a in args.attack_types for r in args.malicious_ratios]

    config_record = asdict(base)
    config_record.update({
        "num_train_examples": len(data.train),
        "num_reference_examples": len(data.reference),
        "num_global_examples": len(data.global_set) if data.global_set is not None else 0,
        "num_test_examples": len(data.test),
        "shapley_eval_tokens": shapley_eval.tokens,
        "partition_cluster_sizes": data.cluster_sizes,
        "axis_num_groups": {g.name: g.num_groups for g in groupings},
        "axis_group_sizes": {g.name: g.counts.tolist() for g in groupings},
        "lora_trainable_parameters": model.num_trainable(),
        "torch_version": torch.__version__,
    })
    check_same_configuration(base)
    write_json(os.path.join(base.results_dir, "experiment_config.json"), config_record)
    ctx = RunContext(model, initial_state, data, collate, shapley_eval, reference_eval,
                     test_eval, groupings, client_grouping)

    for seed in args.experiment_seeds:
        if base.partition_method == "iid":
            partition = iid_partition(data.train, base.num_clients, seed=seed)
        else:
            partition = dirichlet_partition(
                data.partition_clusters, base.num_clients, alpha=base.dirichlet_alpha,
                min_samples_per_client=base.min_samples_per_client, seed=seed)
        stats = partition_frame(data, partition).assign(run_seed=seed)
        stats.to_csv(os.path.join(base.results_dir, f"partition_seed{seed}.csv"), index=False)
        attacker_order = np.random.default_rng(seed).permutation(base.num_clients)
        schedule = build_participation_schedule(
            base.num_clients, base.participation_ratio, base.num_rounds, seed=seed + 1_000_003)
        logger.info("seed %d client examples=%s attacker_order=%s", seed,
                    stats["total_samples"].tolist(), attacker_order.tolist())

        for attack, ratio in attacks:
            name = experiment_name(attack, ratio, part_id)
            run_dir = os.path.join(base.results_dir, "runs", f"{name}_seed{seed}")
            if os.path.exists(os.path.join(run_dir, "summary.json")):
                logger.info("skip completed %s seed=%d", name, seed)
                continue
            os.makedirs(run_dir, exist_ok=True)
            config = copy.deepcopy(base)
            config.seed = seed
            config.partition_seed = seed
            config.experiment_name = name
            config.attack_type = attack
            config.malicious_ratio = ratio
            num_mal = int(round(base.num_clients * ratio)) if attack != "none" else 0
            malicious_ids = set(attacker_order[:num_mal].tolist())
            started = time.perf_counter()
            state = run_experiment(config, ctx, partition, malicious_ids, schedule,
                                   checkpoint_path=os.path.join(run_dir, "checkpoint.pt"))
            finish_run(config, run_dir, state, malicious_ids, groupings[0].num_groups,
                       partition_frame=stats, config_record=config_record,
                       wall_seconds=time.perf_counter() - started,
                       make_plots=not args.no_plots)
            refresh_summary(args.summary_root or base.results_dir, not args.no_plots)

    logger.info("partition %s finished", part_id)


if __name__ == "__main__":
    main()
