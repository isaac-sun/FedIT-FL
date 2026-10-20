from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from config import Config
from contribution.shapley import EvalSet, Grouping, estimate_round_shapley, example_losses
from data.dolly import (
    CATEGORIES,
    encode_rows,
    load_records,
    load_reference_units,
    make_collate,
    make_splits,
    random_labels_like,
)
from fl.aggregation import fedavg_aggregate
from main import pairwise_auc
from models.lora import inject_lora, load_lora_state, lora_state_dict
from tests.conftest import DATA


@pytest.fixture(scope="module")
def frame():
    return load_records(DATA)


def test_original_dolly_file(frame):
    assert len(frame) == 15_014
    assert sorted(frame["category"].unique()) == sorted(CATEGORIES)


def test_splits_are_disjoint_and_reference_is_balanced(frame):
    splits = make_splits(frame, 60, 480, 1000, seed=0)
    sets = {name: set(rows.tolist()) for name, rows in splits.items()}
    assert sum(map(len, sets.values())) == len(frame)
    assert len(set.union(*sets.values())) == len(frame)
    counts = frame.set_index("row_id").loc[splits["reference"], "category"].value_counts()
    assert set(counts) == {60} and len(counts) == 8
    assert (len(sets["global"]), len(sets["test"])) == (480, 1000)
    again = make_splits(frame, 60, 480, 1000, seed=0)
    assert all(np.array_equal(splits[k], again[k]) for k in splits)
    # a smaller global axis leaves every other split unchanged
    smaller = make_splits(frame, 60, 120, 1000, seed=0)
    assert set(smaller["global"]) < set(splits["global"])
    assert all(np.array_equal(splits[k], smaller[k]) for k in ("reference", "test", "train"))


def test_frozen_llm_units_cover_the_reference_set(frame):
    config = Config()
    splits = make_splits(frame, config.reference_per_category, config.global_size,
                         config.test_size, config.split_seed)
    units = load_reference_units(config.llm_units_file, splits["reference"])
    assert units.shape == (480,) and units.min() == 0 and units.max() == 9
    assert np.bincount(units).min() >= 30
    with pytest.raises(SystemExit):
        load_reference_units(config.llm_units_file, np.array([0, 1, 2]))
    shuffled = random_labels_like(units, seed=1)
    assert np.array_equal(np.bincount(shuffled), np.bincount(units))
    assert not np.array_equal(shuffled, units)


def test_encoding_keeps_the_response_and_eos(tiny_model, frame):
    from models.llm import load_tokenizer
    tokenizer = load_tokenizer(tiny_model)
    rows = frame[frame["context"].str.len() > 2000].head(5)
    encoded = encode_rows(tokenizer, rows, max_length=128, max_response_length=32)
    for item in encoded:
        ids, start = item["input_ids"], item["prompt_length"]
        assert len(ids) <= 128 and ids[-1] == tokenizer.eos_token_id
        assert 1 <= len(ids) - start <= 32
        assert tokenizer.decode(ids[:start]).endswith("### Response:\n")
    input_ids, attention, labels, _ = make_collate(tokenizer.pad_token_id)(
        [(e["input_ids"], e["prompt_length"], 0) for e in encoded])
    first = encoded[0]
    assert (labels[0] != -100).sum() == len(first["input_ids"]) - first["prompt_length"]
    assert attention[0].sum() == len(first["input_ids"])


def test_lora_starts_at_the_base_model_and_round_trips():
    torch.manual_seed(0)
    model = nn.Sequential(OrderedDict(q_proj=nn.Linear(8, 8), act=nn.ReLU(), out=nn.Linear(8, 4)))
    x = torch.randn(5, 8)
    before = model(x)
    replaced = inject_lora(model, ("q_proj", "out"), r=2, alpha=4.0, dropout=0.0, seed=0)
    assert replaced == ["q_proj", "out"]
    assert torch.allclose(model(x), before)          # B = 0
    state = lora_state_dict(model)
    assert list(state) == ["q_proj.lora_A", "q_proj.lora_B", "out.lora_A", "out.lora_B"]
    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora" not in n)
    changed = OrderedDict((k, v + 0.1) for k, v in state.items())
    load_lora_state(model, changed)
    assert not torch.allclose(model(x), before)
    assert all(torch.equal(lora_state_dict(model)[k], changed[k]) for k in changed)
    with pytest.raises(KeyError):
        load_lora_state(model, {"q_proj.lora_A": state["q_proj.lora_A"]})
    average = fedavg_aggregate(state, {0: OrderedDict((k, torch.ones_like(v)) for k, v in state.items()),
                                       1: OrderedDict((k, -torch.ones_like(v)) for k, v in state.items())},
                               weights={0: 3, 1: 1})
    assert torch.allclose(average["out.lora_B"], state["out.lora_B"] + 0.5)


def test_shapley_values_add_up_per_group(tiny_model, frame):
    """Every permutation telescopes: sum_i phi_ig = v_g(all) - v_g(none)."""
    from models.llm import create_model, load_tokenizer
    config = Config(model_path=tiny_model, lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
    tokenizer = load_tokenizer(tiny_model)
    model = create_model(config, "cpu")
    rows = frame.sample(24, random_state=0)
    examples = [(e["input_ids"], e["prompt_length"], 0)
                for e in encode_rows(tokenizer, rows, 96, 24)]
    evalset = EvalSet(examples, make_collate(tokenizer.pad_token_id), "cpu", token_budget=600)
    assert len(evalset.batches) > 1
    labels = np.r_[np.arange(16) % 4, np.full(8, -1)]
    groupings = [Grouping("four", labels, 4),
                 Grouping("global", np.r_[np.full(16, -1), np.zeros(8, dtype=int)], 1)]
    global_state = model.get_state_dict()
    rng = torch.Generator().manual_seed(1)
    updates = {cid: OrderedDict((k, 0.05 * torch.randn(v.shape, generator=rng))
                                for k, v in global_state.items()) for cid in range(4)}
    values, evaluations = estimate_round_shapley(
        model, updates, global_state, evalset, groupings, num_mc_samples=5,
        device="cpu", rng=np.random.default_rng(0))
    assert evaluations <= 1 + 5 * 4
    full = OrderedDict((k, v + sum(u[k] for u in updates.values()) / 4)
                       for k, v in global_state.items())
    base = example_losses(model, global_state, evalset, "cpu")
    after = example_losses(model, full, evalset, "cpu")
    for grouping in groupings:
        total = sum(values[grouping.name][cid] for cid in updates)
        np.testing.assert_allclose(total, grouping.means(base) - grouping.means(after),
                                   rtol=1e-5, atol=1e-6)   # float32 summation order
    # losses come back in the original order whatever the batching
    alone = example_losses(model, global_state,
                           EvalSet(examples[:1], make_collate(tokenizer.pad_token_id), "cpu", 600),
                           "cpu")
    np.testing.assert_allclose(alone[0], base[0], rtol=1e-5)


def test_pairwise_auc():
    assert pairwise_auc([3, 4], [1, 2]) == 1.0
    assert pairwise_auc([1, 2], [3, 4]) == 0.0
    assert pairwise_auc([1, 1], [1]) == 0.5


def test_axis_tests_pair_runs_by_condition_and_seed():
    import summarize
    rows = []
    for partition, gain in (("iid", 0.0), ("dirichlet_alpha0p5", 0.05), ("dirichlet_alpha0p1", 0.1)):
        for seed in (1, 2, 3):
            for attack in ("dfr", "afr"):
                base = 0.7 + 0.01 * seed
                for axis, auc in (("llm", base + gain), ("llm_random", base), ("category", base),
                                  ("global", np.nan)):
                    record = {"partition_id": partition, "attack_type": attack,
                              "malicious_ratio": 0.3, "run_seed": seed, "axis": axis,
                              "experiment_name": f"{attack}_{partition}"}
                    for name in summarize.AXIS_SCORES:
                        record[f"auc_{name}"] = auc if name.startswith("V") else 0.6
                    if axis == "global":
                        record.update({f"auc_{n}": 0.65 for n in summarize.AXIS_SCORES
                                       if n.startswith("SV")})
                    rows.append(record)
    tests = summarize.axis_tests(pd.DataFrame(rows))
    h1 = tests[(tests["test"] == "llm_vs_control") & (tests["partition_id"] == "all")
               & (tests["score"] == "V_mean") & (tests["control"] == "llm_random")].iloc[0]
    assert h1["n"] == 18 and h1["wins"] == 12 and h1["ties"] == 6
    assert h1["mean_diff"] == pytest.approx(0.05)
    trend = tests[(tests["test"] == "llm_gap_trend") & (tests["score"] == "V_mean")].iloc[0]
    assert trend["spearman_rho"] > 0.9
    versus = tests[(tests["test"] == "class_V_vs_global_SV") & (tests["axis"] == "llm")
                   & (tests["partition_id"] == "iid") & (tests["score"] == "V_mean")].iloc[0]
    assert versus["mean_auc_control"] == pytest.approx(0.65)
