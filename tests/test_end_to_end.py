"""End-to-end runs of main.py on CPU with a tiny random Qwen2 model."""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import main as fedit_main
from data.dolly import load_records, make_splits
from tests.conftest import DATA, ROOT


def units_file(folder):
    """Random 3-unit labels for the small reference set used here."""
    splits = make_splits(load_records(DATA), 3, 16, 40, seed=0)
    units = np.arange(len(splits["reference"])) % 3
    path = os.path.join(folder, "units.csv")
    pd.DataFrame({"row_id": splits["reference"],
                  "unit_id": np.random.default_rng(0).permutation(units)}).to_csv(path, index=False)
    return path


def args(model, results, units, *extra):
    return ["--model-path", model, "--results-dir", results, "--device", "cpu",
            "--partition-method", "dirichlet", "--dirichlet-alpha", "0.5",
            "--attack-types", "dfr,sdfr,afr", "--malicious-ratios", "0.5",
            "--experiment-seeds", "7", "--num-rounds", "4", "--num-clients", "4",
            "--num-mc-samples", "3", "--local-steps", "2", "--batch-size", "4",
            "--micro-batch-size", "2", "--local-lr", "1e-3", "--max-length", "96",
            "--max-response-length", "24", "--reference-per-category", "3",
            "--global-size", "16", "--test-size", "40", "--min-samples-per-client", "50",
            "--llm-units-file", units, "--torch-threads", "2", *extra]


def comparable(path):
    frame = pd.read_csv(path)
    return frame[[c for c in frame.columns if not c.endswith("seconds")]]


@pytest.fixture(scope="module")
def finished(tiny_model, tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("e2e"))
    units = units_file(folder)
    results = os.path.join(folder, "full")
    fedit_main.main(args(tiny_model, results, units))
    return tiny_model, folder, units, results


def test_every_condition_finishes_with_all_axes(finished):
    _, _, _, results = finished
    summaries = [json.load(open(p)) for p in glob.glob(f"{results}/runs/*/summary.json")]
    assert len(summaries) == 4
    attack = next(s for s in summaries if s["attack_type"] == "afr")
    for axis in ("llm", "llm_random", "category", "random", "cluster"):
        assert 0.0 <= attack[f"auc_V_mean__{axis}"] <= 1.0
    assert "auc_SV_mean__global" in attack and "auc_V_mean__global" not in attack
    assert attack["auc_V_mean"] == attack["auc_V_mean__llm"]   # primary axis = llm
    assert np.isfinite(attack["untrained_test_loss"]) and attack["rounds_completed"] == 4
    run = os.path.dirname(glob.glob(f"{results}/runs/attack_afr*/summary.json")[0])
    axis = pd.read_csv(f"{run}/axis_records.csv")
    assert set(axis["axis"]) == {"llm", "llm_random", "category", "random", "cluster", "global"}
    assert len(axis) == 6 * 4 * 4          # axes x rounds x clients
    details = pd.read_csv(f"{run}/round_details.csv")
    assert details["local_train_loss"][~details["is_malicious"]].notna().all()
    assert len(glob.glob(f"{run}/plots/*.png")) == 7
    assert not os.path.exists(f"{run}/checkpoint.pt")
    book = pd.ExcelFile(f"{results}/fedit_csdd_summary.xlsx")
    for sheet in ("AUC by Condition", "Axis Comparison", "Axis Tests", "Test Loss",
                  "round_shapley_details", "per_class_client_metrics"):
        assert sheet in book.sheet_names
    assert os.path.exists(f"{results}/plots/auc_by_axis.png")
    assert pd.ExcelFile(f"{run}/experiment_results.xlsx").sheet_names[0] == "round_shapley_details"


def test_interrupted_run_resumes_to_the_same_result(finished, monkeypatch):
    model, folder, units, results = finished
    resumed = os.path.join(folder, "resumed")
    calls = {"n": 0}
    original = fedit_main.FLServer.update_global_model

    def interrupt_after_two_rounds(self, state):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return original(self, state)

    monkeypatch.setattr(fedit_main.FLServer, "update_global_model", interrupt_after_two_rounds)
    with pytest.raises(KeyboardInterrupt):
        fedit_main.main(args(model, resumed, units, "--no-plots"))
    run = "baseline_no_attack_partition_dirichlet_alpha0p5_seed7"
    assert os.path.exists(f"{resumed}/runs/{run}/checkpoint.pt")
    monkeypatch.setattr(fedit_main.FLServer, "update_global_model", original)
    fedit_main.main(args(model, resumed, units, "--no-plots"))
    for path in glob.glob(f"{results}/runs/*/round_details.csv"):
        name = os.path.basename(os.path.dirname(path))
        pd.testing.assert_frame_equal(comparable(path),
                                      comparable(f"{resumed}/runs/{name}/round_details.csv"))
    # a finished run is skipped, not rerun
    stamp = os.path.getmtime(f"{resumed}/runs/{run}/summary.json")
    fedit_main.main(args(model, resumed, units, "--no-plots"))
    assert os.path.getmtime(f"{resumed}/runs/{run}/summary.json") == stamp
    # other settings never reuse the folder of finished runs
    with pytest.raises(SystemExit, match="local_lr"):
        fedit_main.main(args(model, resumed, units, "--no-plots", "--local-lr", "2e-3"))


def test_lr_probe_report(finished):
    model, folder, units, _ = finished
    probe = os.path.join(folder, "lr_probe")
    for lr in ("1e-3", "3e-3"):
        fedit_main.main(args(model, f"{probe}/lr_{lr}", units, "--no-shapley", "--no-plots",
                             "--attack-types", "", "--malicious-ratios", "",
                             "--num-rounds", "3", "--local-lr", lr))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "lr_probe_report.py"), probe],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    table = pd.read_csv(f"{probe}/test_loss_by_round.csv", index_col=0)
    assert list(table.columns) == ["lr=1e-3", "lr=3e-3"] and list(table.index) == [-1, 0, 1, 2]
