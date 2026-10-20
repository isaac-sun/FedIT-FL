#!/usr/bin/env python3
"""Combine finished FedIT-FL runs into one workbook.

Reads every ``runs/*/summary.json`` below ``--results-root`` and writes
``fedit_csdd_summary.xlsx`` with these sheets:

* ``AUC by Condition``: mean and sample SD over seeds of each score's AUC
  (V_mean = paper Eq. 20 is the primary score; lower score = free rider);
* ``AUC by Run``: one row per condition+seed;
* ``Seed Macro``: per-seed macro AUC over all attack conditions;
* ``Client Totals``: per-client cumulative and per-participation scores;
* ``Test Loss``: final test loss (response-token cross-entropy) and
  perplexity per run, with the matched no-attack run;
* ``Round Curves``: test loss per round (overall and per Dolly category);
* ``Completeness``: finished runs per partition, diverged ones included;
* ``Diverged Runs``: runs stopped early because the global model diverged;
  they are left out of every AUC and test-error table;
* ``Robustness AUC by Condition`` / ``Robustness AUC by Run``: post-hoc
  aggregations of V, P and SV computed from ``round_details`` (no rerun
  needed), see ``robust_scores``.  They are robustness checks, not the
  primary score;
* ``Axis Comparison`` / ``Axis AUC by Condition`` / ``Axis AUC by Run`` /
  ``Axis Definitions``: multi-axis runs only.  Every score (V_mean, P_mean,
  SV_mean and the post-hoc aggregations) on every Shapley axis of the same
  runs (LLM knowledge units, equal-size random units, native Dolly
  categories, random groups, TF-IDF clusters, global), so the axes are
  compared on identical training;
* ``Axis Tests``: paired tests over attack runs (condition x seed) of the
  pre-registered scores: LLM units vs every control axis (hypothesis H1),
  their gap per partition and its trend with non-IID strength (H2), and each
  class axis's V against the classless SV (global axis and reference rows);
* ``round_shapley_details`` / ``experiment_summary`` / ``per_class_records`` /
  ``per_class_client_metrics`` / ``per_class_global_metrics``: all runs
  concatenated, same sheets and columns as each run's
  ``experiment_results.xlsx`` (20NEWS-FL layout); ``axis_shapley_records``
  and ``axis_client_totals`` hold every axis.

Charts are written to ``<results-root>/plots/`` unless ``--no-plots``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

AUC_COLUMNS = ["auc_V_mean", "auc_V_sum", "auc_P_mean", "auc_P_sum",
               "auc_SV_mean", "auc_SV_sum"]


RUN_TABLES = ("client_totals", "round_curves", "round_details",
              "per_class_records", "per_class_client_metrics",
              "axis_records", "axis_client_totals")
EXCEL_MAX_ROWS = 1_048_575


def load(root: str):
    summaries = []
    tables = {name: [] for name in RUN_TABLES}
    for path in sorted(glob.glob(os.path.join(root, "**", "runs", "*", "summary.json"),
                                 recursive=True)):
        run_dir = os.path.dirname(path)
        with open(path) as handle:
            summaries.append(json.load(handle))
        for name in RUN_TABLES:
            file = os.path.join(run_dir, f"{name}.csv")
            if os.path.exists(file) and os.path.getsize(file) > 1:
                tables[name].append(pd.read_csv(file))
    if not summaries:
        raise SystemExit(f"no finished runs below {root}")
    tables = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
              for name, frames in tables.items()}
    return pd.DataFrame(summaries), tables


# Post-hoc aggregations (robustness checks; the primary score stays V_mean).
# The raw scores of the first rounds are orders of magnitude larger than later
# rounds, so the plain participation mean mostly reflects round 0.  The same
# rank and late aggregations of P and SV show whether what survives them is
# specific to V or comes from the aggregation alone.
LATE_FROM_ROUND = 3   # after the 2 DFR warm-up rounds and round 2
SCORE_COLUMNS = {"V": "class_sv_variance", "P": "positive_class_sv_sum",
                 "SV": "round_shapley_value"}
ROBUST_SCORES = {
    "V_rank_mean": "mean over participations of the within-round percentile rank of V",
    "P_rank_mean": "mean over participations of the within-round percentile rank of P",
    "SV_rank_mean": "mean over participations of the within-round percentile rank of SV",
    "V_late_mean": f"participation mean of V over rounds >= {LATE_FROM_ROUND}",
    "P_late_mean": f"participation mean of P over rounds >= {LATE_FROM_ROUND}",
    "SV_late_mean": f"participation mean of SV over rounds >= {LATE_FROM_ROUND}",
    "log10V_mean": "mean over participations of log10(V)",
}


def add_round_ranks(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
    frame = frame.copy()
    for name, column in SCORE_COLUMNS.items():
        frame[f"{name}_rank"] = frame.groupby(keys + ["round"])[column].rank(pct=True)
    frame["log10_v"] = np.log10(frame["class_sv_variance"].clip(lower=1e-30))
    return frame


def post_hoc_scores(part: pd.DataFrame):
    """Per-client post-hoc scores of one run (and axis), plus the value of a
    client without any (late) participation: the most suspicious one, i.e.
    0 rank, the minimum late score (0 for V and P) and the minimum log10 V."""
    by_client = part.groupby("client_id")
    late = part[part["round"] >= LATE_FROM_ROUND].groupby("client_id")
    scores, fill = {}, {}
    for name, column in SCORE_COLUMNS.items():
        scores[f"{name}_rank_mean"] = by_client[f"{name}_rank"].mean()
        scores[f"{name}_late_mean"] = late[column].mean()
        fill[f"{name}_rank_mean"] = 0.0
        fill[f"{name}_late_mean"] = min(0.0, float(part[column].min()))
    scores["log10V_mean"] = by_client["log10_v"].mean()
    fill["log10V_mean"] = float(part["log10_v"].min())
    return scores, fill


def robust_scores(details: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """AUC of the post-hoc aggregations for every attack run (primary axis).

    Lower score = more suspicious, as for V_mean.  The run's V_mean, P_mean
    and SV_mean AUCs are copied alongside for comparison.
    """
    from main import pairwise_auc

    if not len(details) or "class_sv_variance" not in details:
        return pd.DataFrame()
    keys = ["partition_id", "experiment_name", "run_seed"]
    frame = add_round_ranks(details, keys)
    rows = []
    for _, run in runs[runs["attack_type"] != "none"].iterrows():
        mask = np.ones(len(frame), dtype=bool)
        for key in keys:
            mask &= (frame[key] == run[key]).to_numpy()
        part = frame[mask]
        if not len(part):
            continue
        malicious = {int(c) for c in str(run["malicious_client_ids"]).split(",") if c.strip().isdigit()}
        clients = range(int(run.get("num_clients", part["client_id"].max() + 1)))
        is_mal = np.array([c in malicious for c in clients])
        scores, fill = post_hoc_scores(part)
        record = {k: run[k] for k in ["partition_id", "attack_type", "malicious_ratio",
                                      "run_seed", "experiment_name"]}
        for name in ("V_mean", "P_mean", "SV_mean"):
            record[f"auc_{name}"] = run.get(f"auc_{name}", np.nan)
        for name in ROBUST_SCORES:
            values = scores[name].reindex(clients).fillna(fill[name]).to_numpy()
            record[f"auc_{name}"] = pairwise_auc(values[~is_mal], values[is_mal], True)
        rows.append(record)
    return pd.DataFrame(rows)


AXIS_SCORES = ("V_mean", "P_mean", "SV_mean", "V_rank_mean", "P_rank_mean", "SV_rank_mean",
               "V_late_mean", "P_late_mean", "SV_late_mean", "log10V_mean")
AXIS_DEFINITIONS = {
    "llm": "LLM knowledge units: frozen reference-row-to-unit file written by an LLM from "
           "the instruction, context and response (groups/llm_units/); primary arm",
    "llm_random": "random reference-row-to-unit assignment with the LLM unit sizes "
                  "(size-matched control for the semantic gain of the LLM units)",
    "category": "the 8 native Dolly categories (task types), 60 reference rows each",
    "random": "random reference-row-to-group assignment with the category sizes",
    "cluster": "nearest TF-IDF k-means cluster of the training texts (the clusters "
               "that drive the Dirichlet non-IID partition)",
    "global": "no classes: one group, its own rows at natural proportions (--global-size); "
              "its SV is the plain Shapley value of the loss on them, so V and P are undefined",
}


def axis_scores(axis_records: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """AUC of every score on every Shapley axis; one row per attack run and axis.

    V_mean, P_mean and SV_mean equal the per-run ``auc_*`` of that axis; the
    post-hoc aggregations follow ``robust_scores``.  Lower = more suspicious.
    """
    from main import pairwise_auc

    if not len(axis_records) or "axis" not in axis_records:
        return pd.DataFrame()
    keys = ["partition_id", "experiment_name", "run_seed"]
    frame = add_round_ranks(axis_records, keys + ["axis"])
    order = list(dict.fromkeys(frame["axis"]))
    groups = dict(iter(frame.groupby(keys + ["axis"], sort=False)))
    rows = []
    for _, run in runs[runs["attack_type"] != "none"].iterrows():
        malicious = {int(c) for c in str(run["malicious_client_ids"]).split(",") if c.strip().isdigit()}
        clients = list(range(int(run.get("num_clients", frame["client_id"].max() + 1))))
        is_mal = np.array([c in malicious for c in clients])
        for axis in order:
            part = groups.get((run["partition_id"], run["experiment_name"], run["run_seed"], axis))
            if part is None:
                continue
            scores, fill = post_hoc_scores(part)
            for name, column in SCORE_COLUMNS.items():
                scores[f"{name}_mean"] = part.groupby("client_id")[column].mean()
                fill[f"{name}_mean"] = 0.0
            record = {k: run[k] for k in ["partition_id", "attack_type", "malicious_ratio",
                                          "run_seed", "experiment_name"]}
            record["axis"] = axis
            single_group = int(part["num_groups"].iloc[0]) == 1 if "num_groups" in part else False
            for name in AXIS_SCORES:
                if single_group and not name.startswith("SV"):
                    record[f"auc_{name}"] = np.nan   # one group: no spread over classes
                    continue
                values = scores[name].reindex(clients).fillna(fill[name]).to_numpy()
                record[f"auc_{name}"] = pairwise_auc(values[~is_mal], values[is_mal], True)
            rows.append(record)
    return pd.DataFrame(rows)


def axis_tables(axis_runs: pd.DataFrame):
    """Per-condition means and the side-by-side comparison of the axes."""
    if not len(axis_runs):
        return pd.DataFrame(), pd.DataFrame()
    keys = ["partition_id", "attack_type", "malicious_ratio"]
    order = list(dict.fromkeys(axis_runs["axis"]))
    cols = [f"auc_{name}" for name in AXIS_SCORES]
    grouped = axis_runs.groupby(keys + ["axis"], sort=False)[cols]
    by_condition = grouped.mean().add_suffix("_mean").join(
        grouped.std(ddof=1).add_suffix("_sd")).join(
        grouped.size().rename("num_seeds")).reset_index()
    long = axis_runs.melt(id_vars=keys + ["run_seed", "axis"], value_vars=cols,
                          var_name="score", value_name="auc")
    long["score"] = pd.Categorical(long["score"].str[len("auc_"):], AXIS_SCORES, ordered=True)
    overall = long.pivot_table(index="score", columns="axis", values="auc",
                               aggfunc="mean", observed=True).reset_index()
    overall[keys] = ["all", "all", float("nan")]
    per_condition = long.pivot_table(index=keys + ["score"], columns="axis", values="auc",
                                     aggfunc="mean", observed=True).reset_index()
    comparison = pd.concat([overall, per_condition], ignore_index=True)
    comparison = comparison[keys + ["score"] + [a for a in order if a in comparison]]
    return by_condition, comparison


TEST_SCORES = ("V_mean", "V_rank_mean", "V_late_mean")   # pre-registered
CONTROL_AXES = ("llm_random", "category", "random", "cluster")
CLASS_AXES = ("llm", "llm_random", "category", "random", "cluster")
# Non-IID strength of each partition, for the H2 trend (larger = more skewed).
NONIID_ORDER = {"iid": 0, "dirichlet_alpha1p0": 1, "dirichlet_alpha0p5": 2,
                "dirichlet_alpha0p1": 3}


def paired_test(diff: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test of paired AUC differences."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    record = {"n": int(diff.size), "mean_diff": float(diff.mean()) if diff.size else np.nan,
              "median_diff": float(np.median(diff)) if diff.size else np.nan,
              "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
              "losses": int((diff < 0).sum()), "p_wilcoxon": np.nan}
    if diff.size and np.any(diff != 0):
        try:
            from scipy.stats import wilcoxon
            record["p_wilcoxon"] = float(wilcoxon(diff, zero_method="zsplit").pvalue)
        except Exception:  # scipy missing or too few pairs
            pass
    return record


def axis_tests(axis_runs: pd.DataFrame) -> pd.DataFrame:
    """Paired comparisons of axes on the same attack runs.

    * ``llm_vs_control``: AUC(llm) - AUC(control) of the same score (H1);
      positive = the LLM units separate free riders better.
    * ``llm_gap_trend``: Spearman correlation, over runs, between that gap
      against ``llm_random`` and the non-IID strength of the partition (H2:
      the gap grows as alpha shrinks).
    * ``class_V_vs_global_SV`` / ``class_V_vs_reference_SV``: AUC of a class
      axis's V score minus the AUC of the same aggregation of the classless
      SV, on the global rows or on the reference rows (the SV of every class
      axis over the reference rows is the same number).
    """
    if not len(axis_runs):
        return pd.DataFrame()
    keys = ["partition_id", "attack_type", "malicious_ratio", "run_seed"]
    wide = axis_runs.pivot_table(index=keys, columns="axis",
                                 values=[f"auc_{s}" for s in AXIS_SCORES],
                                 aggfunc="first")
    axes = set(axis_runs["axis"])
    partitions = ["all"] + sorted(set(axis_runs["partition_id"]),
                                  key=lambda p: NONIID_ORDER.get(p, 99))
    rows = []

    def subset(partition):
        if partition == "all":
            return wide
        return wide[wide.index.get_level_values("partition_id") == partition]

    def column(frame, score, axis):
        key = (f"auc_{score}", axis)
        return frame[key].to_numpy(dtype=float) if key in frame else None

    for partition in partitions:
        frame = subset(partition)
        for score in TEST_SCORES:
            llm = column(frame, score, "llm") if "llm" in axes else None
            if llm is None:
                break
            for control in CONTROL_AXES:
                other = column(frame, score, control) if control in axes else None
                if other is None:
                    continue
                rows.append(dict(test="llm_vs_control", partition_id=partition, score=score,
                                 axis="llm", control=control,
                                 mean_auc_axis=float(np.nanmean(llm)),
                                 mean_auc_control=float(np.nanmean(other)),
                                 **paired_test(llm - other)))
        for score in TEST_SCORES:
            sv_score = "SV" + score[1:]          # V_rank_mean -> SV_rank_mean
            for control, label in (("global", "class_V_vs_global_SV"),
                                   ("reference", "class_V_vs_reference_SV")):
                for axis in CLASS_AXES:
                    if axis not in axes:
                        continue
                    if control == "global":
                        other = column(frame, sv_score, "global") if "global" in axes else None
                    else:
                        other = column(frame, sv_score, axis)
                    value = column(frame, score, axis)
                    if other is None or value is None:
                        continue
                    rows.append(dict(test=label, partition_id=partition, score=score,
                                     axis=axis, control=f"{control}:{sv_score}",
                                     mean_auc_axis=float(np.nanmean(value)),
                                     mean_auc_control=float(np.nanmean(other)),
                                     **paired_test(value - other)))
    if {"llm", "llm_random"} <= axes:
        strength = np.array([NONIID_ORDER.get(p, np.nan)
                             for p in wide.index.get_level_values("partition_id")], dtype=float)
        for score in TEST_SCORES:
            gap = column(wide, score, "llm") - column(wide, score, "llm_random")
            ok = np.isfinite(gap) & np.isfinite(strength)
            record = dict(test="llm_gap_trend", partition_id="all", score=score, axis="llm",
                          control="llm_random", n=int(ok.sum()), spearman_rho=np.nan,
                          p_spearman=np.nan)
            if ok.sum() > 2 and len(set(strength[ok])) > 1:
                try:
                    from scipy.stats import spearmanr
                    result = spearmanr(strength[ok], gap[ok])
                    record.update(spearman_rho=float(result.statistic),
                                  p_spearman=float(result.pvalue))
                except Exception:
                    pass
            rows.append(record)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/full_experiment")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    runs, tables = load(args.results_root)
    # Diverged runs (stopped early, see main.py) stay out of every AUC and
    # test-error table; they are listed in "Diverged Runs" and counted in
    # "Completeness".
    runs["diverged"] = (runs["diverged"].fillna(False).astype(bool)
                        if "diverged" in runs else False)
    all_runs, runs = runs, runs[~runs["diverged"]].copy()
    diverged = all_runs[all_runs["diverged"]]
    totals, curves = tables["client_totals"], tables["round_curves"]
    output = args.output or os.path.join(args.results_root, "fedit_csdd_summary.xlsx")

    attacks = runs[runs["attack_type"] != "none"].copy()
    keys = ["partition_id", "attack_type", "malicious_ratio"]
    present = [c for c in AUC_COLUMNS if c in attacks]
    by_condition = pd.DataFrame()
    seed_macro = pd.DataFrame()
    if len(attacks):
        grouped = attacks.groupby(keys)[present]
        by_condition = grouped.mean().add_suffix("_mean").join(
            grouped.std(ddof=1).add_suffix("_sd")).join(
            grouped.size().rename("num_seeds")).reset_index()
        seed_macro = attacks.groupby(["partition_id", "run_seed"])[present].mean().reset_index()

    base = runs[runs["attack_type"] == "none"][
        ["partition_id", "run_seed", "final_test_loss", "final_test_ppl"]
    ].rename(columns={"final_test_loss": "no_attack_test_loss",
                      "final_test_ppl": "no_attack_test_ppl"})
    test_error = runs[keys + ["run_seed", "untrained_test_loss", "final_test_loss",
                              "final_test_ppl", "best_test_loss",
                              "round0_share_of_loss_drop"]].merge(
        base, on=["partition_id", "run_seed"], how="left")
    test_error["loss_increase_vs_no_attack"] = (
        test_error["final_test_loss"] - test_error["no_attack_test_loss"])
    robust_runs = robust_scores(tables["round_details"], runs)
    robust_condition = pd.DataFrame()
    if len(robust_runs):
        cols = ["auc_V_mean", "auc_P_mean", "auc_SV_mean"] + [f"auc_{n}" for n in ROBUST_SCORES]
        grouped = robust_runs.groupby(keys)[cols]
        robust_condition = grouped.mean().add_suffix("_mean").join(
            grouped.std(ddof=1).add_suffix("_sd")).join(
            grouped.size().rename("num_seeds")).reset_index()
    axis_runs = axis_scores(tables["axis_records"], runs)
    axis_condition, axis_comparison = axis_tables(axis_runs)
    tests = axis_tests(axis_runs)
    completeness = all_runs.groupby("partition_id").agg(
        finished_runs=("experiment_name", "size"),
        diverged_runs=("diverged", "sum"),
        seeds=("run_seed", lambda s: ",".join(map(str, sorted(set(s)))))).reset_index()
    diverged_cols = [c for c in ("partition_id", "attack_type", "malicious_ratio", "run_seed",
                                 "experiment_name", "diverged_round", "rounds_completed",
                                 "best_test_loss", "final_test_loss", "diverged_reason")
                     if c in diverged]
    if len(diverged):
        print("diverged runs (stopped early, left out of the AUC tables):")
        print(diverged[["partition_id", "experiment_name", "run_seed", "diverged_round"]]
              .to_string(index=False))

    tmp = f"{output}.{os.getpid()}.tmp.xlsx"
    with pd.ExcelWriter(tmp) as writer:
        by_condition.to_excel(writer, sheet_name="AUC by Condition", index=False)
        runs.sort_values(keys + ["run_seed"]).to_excel(writer, sheet_name="AUC by Run", index=False)
        seed_macro.to_excel(writer, sheet_name="Seed Macro", index=False)
        totals.to_excel(writer, sheet_name="Client Totals", index=False)
        test_error.to_excel(writer, sheet_name="Test Loss", index=False)
        curves.to_excel(writer, sheet_name="Round Curves", index=False)
        completeness.to_excel(writer, sheet_name="Completeness", index=False)
        if len(diverged):
            diverged[diverged_cols].to_excel(writer, sheet_name="Diverged Runs", index=False)
        robust_condition.to_excel(writer, sheet_name="Robustness AUC by Condition", index=False)
        robust_runs.to_excel(writer, sheet_name="Robustness AUC by Run", index=False)
        pd.DataFrame([{"score": k, "definition": v} for k, v in ROBUST_SCORES.items()]).to_excel(
            writer, sheet_name="Robustness Definitions", index=False)
        if len(axis_runs):
            axis_comparison.to_excel(writer, sheet_name="Axis Comparison", index=False)
            axis_condition.to_excel(writer, sheet_name="Axis AUC by Condition", index=False)
            axis_runs.to_excel(writer, sheet_name="Axis AUC by Run", index=False)
            tests.to_excel(writer, sheet_name="Axis Tests", index=False)
            pd.DataFrame([{"axis": a, "definition": AXIS_DEFINITIONS.get(a, "")}
                          for a in dict.fromkeys(axis_runs["axis"])]).to_excel(
                writer, sheet_name="Axis Definitions", index=False)
        all_runs.to_excel(writer, sheet_name="experiment_summary", index=False)
        for sheet, name in (("round_shapley_details", "round_details"),
                            ("per_class_records", "per_class_records"),
                            ("per_class_client_metrics", "per_class_client_metrics"),
                            ("per_class_global_metrics", "round_curves"),
                            ("axis_shapley_records", "axis_records"),
                            ("axis_client_totals", "axis_client_totals")):
            frame = tables[name]
            if not len(frame) and name.startswith("axis_"):
                continue   # single-axis runs have no axis tables
            if len(frame) > EXCEL_MAX_ROWS:   # keep the full table as CSV
                frame.to_csv(os.path.join(args.results_root, f"all_{name}.csv"), index=False)
                frame = frame.head(EXCEL_MAX_ROWS)
            frame.to_excel(writer, sheet_name=sheet, index=False)
    os.replace(tmp, output)
    if not args.no_plots:
        try:
            from visualization.plots import plot_summary
            plot_summary(args.results_root, runs, robust_runs, axis_runs)
        except Exception as exc:  # charts are optional
            print(f"plotting failed: {exc}")
    print(f"wrote {output} ({len(all_runs)} runs, {len(diverged)} diverged)")
    if len(by_condition):
        cols = keys + [c for c in ("auc_V_mean_mean", "auc_P_mean_mean", "auc_SV_mean_mean")
                       if c in by_condition]
        print(by_condition[cols].round(3).to_string(index=False))
    if len(axis_comparison):
        print("axis comparison, all attack conditions (mean AUC):")
        print(axis_comparison[axis_comparison["partition_id"] == "all"]
              .drop(columns=keys).round(3).to_string(index=False))
    if len(tests):
        print("LLM units vs controls (paired over attack runs, all partitions):")
        part = tests[(tests["test"] == "llm_vs_control") & (tests["partition_id"] == "all")]
        print(part[["score", "control", "n", "mean_diff", "wins", "losses", "p_wilcoxon"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
