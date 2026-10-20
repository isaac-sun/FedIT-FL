#!/usr/bin/env python3
"""Print the main tables of a combined workbook (used by run_fedit.ipynb).

Usage: python3 show_summary.py results/pilot   (or results/full)
"""

import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results/pilot"
path = os.path.join(root, "fedit_csdd_summary.xlsx")
if not os.path.exists(path):
    sys.exit(f"{path} not found (no finished runs yet?)")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
sheets = pd.ExcelFile(path).sheet_names


def show(title, frame):
    print(f"== {title}")
    print(frame.round(3).to_string(index=False) if len(frame) else "(empty)")
    print()


show("Completeness", pd.read_excel(path, "Completeness"))
if "Axis Comparison" in sheets:   # LLM units vs control groupings on the same runs
    axes = pd.read_excel(path, "Axis Comparison")
    show("Axis Comparison, all attack conditions (mean AUC; columns = grouping, first = primary)",
         axes[axes["partition_id"] == "all"].drop(
             columns=["partition_id", "attack_type", "malicious_ratio"]))
    show("Axis Comparison, V_mean by condition",
         axes[(axes["partition_id"] != "all") & (axes["score"] == "V_mean")].drop(columns="score"))
if "Axis Tests" in sheets:
    tests = pd.read_excel(path, "Axis Tests")
    cols = [c for c in ("partition_id", "score", "axis", "control", "n", "mean_auc_axis",
                        "mean_auc_control", "mean_diff", "wins", "losses", "p_wilcoxon",
                        "spearman_rho", "p_spearman") if c in tests]
    show("Axis Tests: LLM units vs controls (paired over attack runs)",
         tests[tests["test"] == "llm_vs_control"][cols])
    show("Axis Tests: H2 trend of AUC(llm) - AUC(llm_random) with non-IID strength",
         tests[tests["test"] == "llm_gap_trend"][cols].dropna(axis=1, how="all"))
    show("Axis Tests: class V vs classless SV (all partitions)",
         tests[tests["test"].str.startswith("class_V") & (tests["partition_id"] == "all")][cols])
auc = pd.read_excel(path, "AUC by Condition")
show("AUC by Condition, primary axis (mean over seeds)",
     auc[[c for c in auc.columns if c in ("partition_id", "attack_type", "malicious_ratio",
                                          "num_seeds") or c.endswith("_mean_mean")]])
if "Robustness AUC by Condition" in sheets:
    robust = pd.read_excel(path, "Robustness AUC by Condition")
    show("Robustness AUC by Condition (post hoc; mean over seeds)",
         robust[[c for c in robust.columns if not c.endswith("_sd")]])
loss = pd.read_excel(path, "Test Loss")
show("Final test loss", loss.groupby(["partition_id", "attack_type", "malicious_ratio"])
     [["untrained_test_loss", "final_test_loss", "loss_increase_vs_no_attack",
       "round0_share_of_loss_drop"]].mean().reset_index())
