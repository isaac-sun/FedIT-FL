#!/usr/bin/env python3
"""Report of ``run_lr_probe.sh``: test loss by round for every learning rate.

Usage: python3 lr_probe_report.py [results/lr_probe]

Writes ``test_loss_by_round.csv``, ``lr_probe_summary.csv`` and
``lr_probe.png`` next to the probe runs.  Round -1 is the untrained base
model.  The suggested rate is the one with the lowest final test loss among
the rates that are still improving or flat at the end (final loss within
0.5% of their best) and whose round 0 takes less than half of the total
loss drop, so the Shapley values of later rounds are not negligible.
"""

import glob
import json
import os
import re
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results/lr_probe"
curves, rows = {}, []
for path in glob.glob(os.path.join(root, "lr_*", "runs", "*", "round_curves.csv")):
    lr = re.search(r"lr_([^/\\]+)", os.path.relpath(path, root)).group(1)
    curve = pd.read_csv(path).set_index("round")["test_loss"]
    summary_path = os.path.join(os.path.dirname(path), "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    untrained = summary.get("untrained_test_loss")
    if untrained is not None:
        curve = pd.concat([pd.Series({-1: untrained}), curve])
    curves[f"lr={lr}"] = curve
    trained = curve.drop(index=-1, errors="ignore")
    final, best = float(trained.iloc[-1]), float(trained.min())
    rows.append({
        "lr": lr, "lr_value": float(lr), "rounds": int(len(trained)),
        "untrained_loss": untrained, "loss_after_round0": float(trained.iloc[0]),
        "final_loss": final, "best_loss": best, "best_round": int(trained.idxmin()),
        "round0_share_of_drop": summary.get("round0_share_of_loss_drop"),
        "drop_last_5_rounds": float(trained.iloc[-6] - final) if len(trained) > 5 else float("nan"),
        "diverged": bool(summary.get("diverged", False)),
        "finished": bool(summary),
    })
if not rows:
    sys.exit(f"no probe runs below {root}")
table = pd.DataFrame(curves).sort_index()
table = table[sorted(table.columns, key=lambda c: float(c[3:]))]
table.to_csv(os.path.join(root, "test_loss_by_round.csv"))
summary = pd.DataFrame(rows).sort_values("lr_value").drop(columns="lr_value")
summary.to_csv(os.path.join(root, "lr_probe_summary.csv"), index=False)
pd.set_option("display.width", 200)
print("test loss by round (round -1 = untrained base model):")
print(table.round(4).to_string())
print()
print(summary.round(4).to_string(index=False))
ok = summary[summary["finished"] & ~summary["diverged"]
             & (summary["final_loss"] <= summary["best_loss"] * 1.005)
             & (summary["round0_share_of_drop"].fillna(1.0) < 0.5)]
if len(ok):
    pick = ok.sort_values("final_loss").iloc[0]
    print(f"\nsuggested LOCAL_LR={pick['lr']} (final loss {pick['final_loss']:.4f}, "
          f"round 0 = {pick['round0_share_of_drop']:.0%} of the loss drop)")
else:
    print("\nno rate meets the rule (still improving at the end and round 0 < 50% of the "
          "drop); pick by hand from the tables above")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column in table:
        ax.plot(table.index, table[column], marker="o", ms=3, label=column)
    ax.set_xlabel("round (-1 = base model)")
    ax.set_ylabel("test loss (response cross-entropy)")
    ax.set_title("learning-rate probe (no attack, Dirichlet alpha=0.5)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(root, "lr_probe.png"), dpi=140)
    print(f"chart: {os.path.join(root, 'lr_probe.png')}")
except Exception as exc:  # the chart is optional
    print(f"plotting failed: {exc}")
