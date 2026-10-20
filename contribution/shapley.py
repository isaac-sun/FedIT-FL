"""Per-group Monte Carlo Shapley values for federated instruction tuning (CSDD).

Adapted from ECL-FL / 20NEWS-FL ``contribution/shapley.py``.  The permutation
sampler, the coalition construction (equal weights, paper Eq. 14) and the
class metrics are the same.  The value function of group ``g`` is

    v_g(S) = F_g(w_t, D_ref) - F_g(w_{t,S}, D_ref)

with ``F_g`` the mean, over the reference examples of group ``g``, of each
example's mean response-token cross-entropy.  One forward pass of a
coalition's model gives the loss of every evaluation example; every grouping
("axis") is then a different average of the same losses.  So all class axes
share the permutations, the coalition models and the evaluation, and each
extra class axis costs nothing.  The "global" axis has its own rows (natural
proportions) appended to the same evaluation set.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch


class EvalSet:
    """Examples padded into length-sorted batches, kept on the device.

    ``batches`` holds ``(input_ids, attention, labels, positions)``, where
    ``positions`` are the examples' indices in the original order; the
    per-example losses come back in that order.
    """

    def __init__(self, examples: list, collate, device: str, token_budget: int):
        self.size = len(examples)
        lengths = np.array([len(item[0]) for item in examples])
        order = np.argsort(-lengths, kind="stable")
        self.batches = []
        start = 0
        while start < len(order):
            stop = start + 1
            width = lengths[order[start]]
            while stop < len(order) and (stop - start + 1) * width <= token_budget:
                stop += 1
            index = order[start:stop]
            input_ids, attention, labels, _ = collate([examples[i] for i in index])
            self.batches.append((input_ids.to(device), attention.to(device),
                                 labels.to(device), torch.as_tensor(index)))
            start = stop
        self.tokens = int(lengths.sum())


@torch.no_grad()
def example_losses(model, state_dict, evalset: EvalSet, device: str) -> np.ndarray:
    """Loss of every example of ``evalset`` under ``state_dict`` (LoRA)."""
    model.load_state_dict_from(state_dict)
    model.eval()
    use_amp = device == "cuda"
    losses = np.empty(evalset.size, dtype=np.float64)
    for input_ids, attention, labels, positions in evalset.batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            values = model.response_losses(input_ids, attention, labels)
        if not torch.isfinite(values).all():
            raise FloatingPointError("non-finite loss during evaluation")
        losses[positions.numpy()] = values.double().cpu().numpy()
    return losses


@dataclass
class Grouping:
    """One definition of the "classes": a group label per evaluation example
    (-1 = not part of this axis)."""
    name: str
    labels: np.ndarray
    num_groups: int

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=np.int64)
        member = self.labels >= 0
        self.counts = np.bincount(self.labels[member], minlength=self.num_groups)
        if (self.counts == 0).any():
            raise ValueError(f"axis {self.name}: empty group(s) {np.flatnonzero(self.counts == 0)}")
        # Overall SV = example-weighted mean over the groups = Shapley value
        # of the mean loss over this axis's rows.
        self.weights = self.counts / self.counts.sum()

    def means(self, losses: np.ndarray) -> np.ndarray:
        member = self.labels >= 0
        sums = np.bincount(self.labels[member], weights=losses[member],
                           minlength=self.num_groups)
        return sums / self.counts


def _coalition_state(global_state, weighted_delta, total_weight):
    state = OrderedDict()
    for key, value in global_state.items():
        state[key] = value + weighted_delta[key] / total_weight
        if not torch.isfinite(state[key]).all():
            raise FloatingPointError(f"non-finite coalition parameter: {key}")
    return state


def estimate_round_shapley(
    model,
    updates: dict,
    global_state_dict: OrderedDict,
    evalset: EvalSet,
    groupings: list,
    *,
    num_mc_samples: int = 20,
    device: str = "cpu",
    sample_counts: dict | None = None,
    cache_coalitions: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[dict, int]:
    """Return ``{axis: {client_id: ndarray[num_groups]}}`` and the number of
    coalition evaluations.  ``sample_counts=None`` = equal-weight coalitions."""
    client_ids = list(updates.keys())
    if not client_ids:
        return {g.name: {} for g in groupings}, 0
    rng = rng if rng is not None else np.random.default_rng()
    weights = sample_counts or {cid: 1 for cid in client_ids}
    base = example_losses(model, global_state_dict, evalset, device)
    evaluations = 1
    cache = {(): base} if cache_coalitions else None
    sums = {g.name: {cid: np.zeros(g.num_groups) for cid in client_ids} for g in groupings}
    base_means = {g.name: g.means(base) for g in groupings}
    for _ in range(num_mc_samples):
        order = rng.permutation(client_ids).tolist()
        coalition = []
        previous = dict(base_means)
        weighted_delta = OrderedDict(
            (key, torch.zeros_like(value)) for key, value in global_state_dict.items())
        total_weight = 0.0
        for cid in order:
            coalition.append(cid)
            weight = float(weights.get(cid, 1))
            total_weight += weight
            for key, accumulator in weighted_delta.items():
                accumulator.add_(updates[cid][key], alpha=weight)
            key = tuple(sorted(coalition))
            losses = cache.get(key) if cache is not None else None
            if losses is None:
                losses = example_losses(
                    model, _coalition_state(global_state_dict, weighted_delta, total_weight),
                    evalset, device)
                evaluations += 1
                if cache is not None:
                    cache[key] = losses
            for grouping in groupings:
                current = grouping.means(losses)
                sums[grouping.name][cid] += previous[grouping.name] - current
                previous[grouping.name] = current
    result = {name: {cid: value / num_mc_samples for cid, value in per_client.items()}
              for name, per_client in sums.items()}
    return result, evaluations


def per_group_to_overall(per_group_sv: dict, group_weights: np.ndarray) -> dict:
    return {cid: float(np.dot(group_weights, sv)) for cid, sv in per_group_sv.items()}


def compute_class_metrics(per_class_sv: dict, eps: float = 1e-10) -> dict:
    """Unchanged from 20NEWS-FL: A (positive sum), P, V, dominant class."""
    metrics = {}
    for cid, sv_arr in per_class_sv.items():
        positive = np.maximum(sv_arr, 0.0)
        a_i = float(np.sum(positive))
        p_i = float(np.max(positive) / (a_i + eps)) if a_i > 0 else 0.0
        v_i = float(np.var(sv_arr))
        c_i = int(np.argmax(positive))
        metrics[cid] = {
            "class_sv_variance": v_i,
            "positive_class_sv_sum": a_i,
            "concentration_ratio": p_i,
            "dominant_class": c_i,
        }
    return metrics
