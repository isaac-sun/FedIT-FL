"""Charts for one run (``plot_run``) and for a results tree (``plot_summary``).

Per-run charts go to ``<run_dir>/plots/``; combined charts to
``<results_root>/plots/``.  Honest clients are blue, free riders red.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CATEGORIES = ("brainstorming", "classification", "closed_qa", "creative_writing",
              "general_qa", "information_extraction", "open_qa", "summarization")
HONEST = "#2f6fb0"
MALICIOUS = "#c8402f"
GROUP_COLORS = plt.get_cmap("tab10").colors


def _save(fig, folder: str, name: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _class_columns(frame: pd.DataFrame, suffix: str = "") -> list[str]:
    columns = [c for c in frame.columns
               if c.startswith("class_") and c.endswith(suffix)
               and c[len("class_"):len(c) - len(suffix)].isdigit()]
    return sorted(columns, key=lambda c: int(c[len("class_"):len(c) - len(suffix)]))


def _group_columns(frame: pd.DataFrame, prefix: str, suffix: str) -> list[str]:
    columns = [c for c in frame.columns if c.startswith(prefix) and c.endswith(suffix)]
    return sorted(columns, key=lambda c: int(c[len(prefix):-len(suffix)]))


def _honest_vs_malicious(ax, frame, column, title, ylabel):
    for flag, color, label in ((False, HONEST, "honest"), (True, MALICIOUS, "free rider")):
        part = frame[frame["is_malicious"] == flag]
        if not len(part):
            continue
        stats = part.groupby("round")[column].agg(["mean", "std"]).reset_index()
        ax.plot(stats["round"], stats["mean"], color=color, marker="o", ms=3, label=label)
        ax.fill_between(stats["round"], stats["mean"] - stats["std"].fillna(0),
                        stats["mean"] + stats["std"].fillna(0), color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def _client_heatmap(ax, matrix, title, malicious, xlabel, cmap="RdBu_r", center=True):
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    kwargs = dict(vmin=-limit, vmax=limit) if center else {}
    image = ax.imshow(values, aspect="auto", cmap=cmap, **kwargs)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{c}{' *' if c in malicious else ''}" for c in matrix.index],
                       fontsize=7)
    for tick, cid in zip(ax.get_yticklabels(), matrix.index):
        if cid in malicious:
            tick.set_color(MALICIOUS)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("client (* = free rider)")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046)


def plot_run(run_dir: str, tables: dict, summary: dict) -> list[str]:
    folder = os.path.join(run_dir, "plots")
    details = tables["round_details"]
    curves = tables["round_curves"]
    per_class = tables["per_class_records"]
    client_metrics = tables["per_class_client_metrics"]
    totals = tables["client_totals"]
    malicious = set(totals.loc[totals["is_malicious"], "client_id"].tolist())
    title = f"{summary['experiment_name']}  seed {summary['run_seed']}"
    paths = []

    # 1. Global test loss and per-category test loss.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    untrained = summary.get("untrained_test_loss")
    rounds = curves["round"].tolist()
    losses = curves["test_loss"].tolist()
    if untrained is not None and np.isfinite(untrained):
        rounds, losses = [-1] + rounds, [untrained] + losses   # -1 = untrained base model
    axes[0].plot(rounds, losses, marker="o", ms=3, label="test loss")
    axes[0].set_title("global model: test loss (round -1 = base model)")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("response cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for i, column in enumerate(_group_columns(curves, "test_group_", "_loss")):
        g = int(column[len("test_group_"):-len("_loss")])
        name = CATEGORIES[g] if g < len(CATEGORIES) else column
        axes[1].plot(curves["round"], curves[column], marker="o", ms=3,
                     color=GROUP_COLORS[i % 10], label=name)
    axes[1].set_title("global model: test loss per Dolly category")
    axes[1].set_xlabel("round")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    paths.append(_save(fig, folder, "01_test_loss_curves.png"))

    if len(per_class):
        # 2. Shapley values per round, honest vs free rider.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        _honest_vs_malicious(axes[0], details, "round_shapley_value",
                             "round Shapley value", "SV")
        _honest_vs_malicious(axes[1], details, "class_sv_variance",
                             "V: variance of per-class SV", "V")
        _honest_vs_malicious(axes[2], details, "positive_class_sv_sum",
                             "P: sum of positive per-class SV", "P")
        fig.suptitle(title)
        paths.append(_save(fig, folder, "02_shapley_metrics_by_round.png"))

        # 3. Client x round heatmap of the round Shapley value.
        pivot = details.pivot_table(index="client_id", columns="round",
                                    values="round_shapley_value")
        fig, ax = plt.subplots(figsize=(12, 6))
        _client_heatmap(ax, pivot, f"round Shapley value (blank = not selected)\n{title}",
                        malicious, "round")
        paths.append(_save(fig, folder, "03_shapley_heatmap.png"))

        # 4. Mean per-class Shapley profile.
        classes = _class_columns(per_class)
        profile = per_class.groupby("client_id")[classes].mean()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 gridspec_kw={"width_ratios": [1.2, 1]})
        _client_heatmap(axes[0], profile, "mean per-class Shapley value", malicious, "class")
        width = 0.38
        x = np.arange(len(classes))
        for offset, flag, color, label in ((-width / 2, False, HONEST, "honest"),
                                           (width / 2, True, MALICIOUS, "free rider")):
            part = per_class[per_class["is_malicious"] == flag]
            if len(part):
                axes[1].bar(x + offset, part[classes].mean(), width, yerr=part[classes].std(),
                            color=color, alpha=0.85, capsize=2, label=label)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, fontsize=8)
        axes[1].axhline(0, color="black", lw=0.6)
        axes[1].set_title("per-class SV: honest vs free rider (mean ± sd)")
        axes[1].legend()
        fig.suptitle(title)
        paths.append(_save(fig, folder, "04_per_class_sv_profile.png"))

        # 5. Client-level detection scores.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        colors = [MALICIOUS if m else HONEST for m in totals["is_malicious"]]
        for ax, column, label in zip(axes, ("v_mean", "p_mean", "sv_mean"),
                                     ("V mean (Eq. 20, primary)", "P mean", "SV mean")):
            ax.bar(totals["client_id"], totals[column], color=colors)
            auc = summary.get(f"auc_{label.split()[0]}_mean")
            ax.set_title(label + (f"   AUC={auc:.3f}" if auc is not None else ""))
            ax.set_xlabel("client")
            ax.set_xticks(totals["client_id"])
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle(f"{title}   (red = free rider; lower = more suspicious)")
        paths.append(_save(fig, folder, "05_client_scores.png"))

        # 6. V vs SV scatter.
        fig, ax = plt.subplots(figsize=(6, 5))
        seen = totals[totals["participations"] > 0]
        for flag, color, label in ((False, HONEST, "honest"), (True, MALICIOUS, "free rider")):
            part = seen[seen["is_malicious"] == flag]
            ax.scatter(part["sv_mean"], part["v_mean"], color=color, label=label, s=40)
            for _, row in part.iterrows():
                ax.annotate(int(row["client_id"]), (row["sv_mean"], row["v_mean"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("SV mean")
        ax.set_ylabel("V mean")
        ax.set_title(title, fontsize=9)
        ax.legend()
        ax.grid(alpha=0.3)
        paths.append(_save(fig, folder, "06_v_vs_sv_scatter.png"))

    # 7. Per-class loss of each client's submitted model on the reference set.
    loss_columns = _class_columns(client_metrics, "_loss")
    if loss_columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 gridspec_kw={"width_ratios": [1.2, 1]})
        matrix = client_metrics.groupby("client_id")[loss_columns].mean()
        matrix.columns = [c[:-len("_loss")] for c in loss_columns]
        _client_heatmap(axes[0], matrix, "local model: mean reference loss per class",
                        malicious, "class", cmap="viridis", center=False)
        _honest_vs_malicious(axes[1], details, "local_ref_loss",
                             "local model: reference loss by round", "loss")
        fig.suptitle(title)
        paths.append(_save(fig, folder, "07_client_class_loss.png"))
    return paths


# ── combined charts ─────────────────────────────────────────────────────────

def plot_summary(root: str, runs: pd.DataFrame, robust: pd.DataFrame | None = None,
                 axis_runs: pd.DataFrame | None = None) -> list[str]:
    folder = os.path.join(root, "plots")
    paths = []
    attacks = runs[runs["attack_type"] != "none"].copy()
    if len(attacks) and "auc_V_mean" in attacks:
        attacks["condition"] = (attacks["attack_type"].str.upper() + " "
                                + (attacks["malicious_ratio"] * 100).round().astype(int).astype(str) + "%")
        order = sorted(attacks["condition"].unique(),
                       key=lambda c: (c.split()[0], int(c.split()[1][:-1])))

        # AUC heatmap per score.
        scores = [c for c in ("auc_V_mean", "auc_P_mean", "auc_SV_mean") if c in attacks]
        fig, axes = plt.subplots(1, len(scores), figsize=(5.5 * len(scores),
                                                          1.2 + 0.6 * attacks["partition_id"].nunique()))
        axes = np.atleast_1d(axes)
        for ax, score in zip(axes, scores):
            table = attacks.pivot_table(index="partition_id", columns="condition",
                                        values=score, aggfunc="mean")[order]
            image = ax.imshow(table.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1,
                              aspect="auto")
            for (i, j), value in np.ndenumerate(table.to_numpy(dtype=float)):
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(table.index)))
            ax.set_yticklabels(table.index, fontsize=8)
            ax.set_title(f"{score[4:]} AUC (mean over seeds)")
            plt.colorbar(image, ax=ax, fraction=0.046)
        paths.append(_save(fig, folder, "auc_heatmap.png"))

        # AUC per condition with seed spread, primary score.
        fig, ax = plt.subplots(figsize=(12, 4.5))
        partitions = sorted(attacks["partition_id"].unique())
        width = 0.8 / max(len(partitions), 1)
        x = np.arange(len(order))
        for k, part in enumerate(partitions):
            sub = attacks[attacks["partition_id"] == part].groupby("condition")["auc_V_mean"]
            mean = sub.mean().reindex(order)
            sd = sub.std().reindex(order).fillna(0)
            ax.bar(x + (k - (len(partitions) - 1) / 2) * width, mean, width, yerr=sd,
                   capsize=2, label=part, color=GROUP_COLORS[k % 10])
        ax.axhline(0.5, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC (V mean)")
        ax.set_title("CSDD detection AUC by condition (mean ± sd over seeds)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        paths.append(_save(fig, folder, "auc_by_condition.png"))

    # Final test loss by condition.
    if len(runs):
        frame = runs.copy()
        frame["condition"] = np.where(
            frame["attack_type"] == "none", "no attack",
            frame["attack_type"].str.upper() + " "
            + (frame["malicious_ratio"] * 100).round().astype(int).astype(str) + "%")
        table = frame.pivot_table(index="condition", columns="partition_id",
                                  values="final_test_loss", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        table.plot.bar(ax=ax, color=GROUP_COLORS[:table.shape[1]])
        ax.set_ylabel("final test loss")
        ax.set_title("final global test loss by condition (mean over seeds)")
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        paths.append(_save(fig, folder, "final_test_loss.png"))

    # Primary V_mean vs the post-hoc V aggregations (robustness check).
    if robust is not None and len(robust):
        frame = robust.copy()
        frame["condition"] = (frame["partition_id"] + "\n" + frame["attack_type"].str.upper() + " "
                              + (frame["malicious_ratio"] * 100).round().astype(int).astype(str) + "%")
        scores = ["auc_V_mean", "auc_V_rank_mean", "auc_log10V_mean", "auc_V_late_mean"]
        table = frame.groupby("condition")[scores].mean()
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(table)), 4.8))
        width = 0.8 / len(scores)
        x = np.arange(len(table))
        for k, score in enumerate(scores):
            ax.bar(x + (k - 1.5) * width, table[score], width, color=GROUP_COLORS[k],
                   label=score[4:] + (" (primary)" if score == "auc_V_mean" else " (post hoc)"))
        ax.axhline(0.5, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(table.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC (mean over seeds)")
        ax.set_title("V aggregations: primary vs robustness checks")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.3, axis="y")
        paths.append(_save(fig, folder, "auc_v_robustness.png"))

        # V vs P vs SV under each aggregation, one panel per partition: does
        # the signal that survives removing the round scale belong to V only?
        aggregations = [("participation mean", "_mean"), ("within-round rank", "_rank_mean"),
                        ("late rounds", "_late_mean")]
        names = [n for n in ("V", "P", "SV")
                 if all(f"auc_{n}{suffix}" in frame for _, suffix in aggregations)]
        partitions = sorted(frame["partition_id"].unique())
        if names:
            fig, axes = plt.subplots(1, len(partitions), figsize=(3.6 * len(partitions) + 1.5, 4.2),
                                     sharey=True, squeeze=False)
            width = 0.8 / len(names)
            x = np.arange(len(aggregations))
            for ax, part in zip(axes[0], partitions):
                sub = frame[frame["partition_id"] == part]
                for k, name in enumerate(names):
                    cols = [f"auc_{name}{suffix}" for _, suffix in aggregations]
                    ax.bar(x + (k - (len(names) - 1) / 2) * width, sub[cols].mean(), width,
                           yerr=sub[cols].std().fillna(0), capsize=2, color=GROUP_COLORS[k],
                           label=name)
                ax.axhline(0.5, color="black", lw=0.8, ls="--")
                ax.set_xticks(x)
                ax.set_xticklabels([label for label, _ in aggregations], rotation=20, ha="right",
                                   fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.set_title(part, fontsize=9)
                ax.grid(alpha=0.3, axis="y")
            axes[0, 0].set_ylabel("AUC (mean ± sd over attack runs)")
            axes[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
            fig.suptitle("primary grouping: V vs P vs SV under each aggregation")
            paths.append(_save(fig, folder, "auc_v_p_sv.png"))

    # Multi-axis runs: every score on every grouping of the same client updates.
    if axis_runs is not None and len(axis_runs):
        axes_order = list(dict.fromkeys(axis_runs["axis"]))
        scores = [c for c in ("auc_V_mean", "auc_P_mean", "auc_SV_mean", "auc_V_rank_mean",
                              "auc_P_rank_mean", "auc_SV_rank_mean", "auc_V_late_mean",
                              "auc_P_late_mean", "auc_SV_late_mean", "auc_log10V_mean")
                  if c in axis_runs]
        mean = axis_runs.groupby("axis")[scores].mean().reindex(axes_order)
        sd = axis_runs.groupby("axis")[scores].std().reindex(axes_order).fillna(0)
        fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(scores)), 4.8))
        width = 0.8 / len(axes_order)
        x = np.arange(len(scores))
        for k, axis in enumerate(axes_order):
            ax.bar(x + (k - (len(axes_order) - 1) / 2) * width, mean.loc[axis], width,
                   yerr=sd.loc[axis], capsize=2, color=GROUP_COLORS[k % 10],
                   label=axis + (" (primary)" if k == 0 else ""))
        ax.axhline(0.5, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([s[4:] for s in scores], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC (mean ± sd over attack runs)")
        ax.set_title("detection AUC by grouping axis (same training runs)")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.3, axis="y")
        paths.append(_save(fig, folder, "auc_by_axis.png"))

        # H2: the gain of the LLM units over size-matched random units per
        # partition, ordered from IID to the strongest non-IID.
        if {"llm", "llm_random"} <= set(axes_order):
            keys = ["partition_id", "attack_type", "malicious_ratio", "run_seed"]
            order = {"iid": 0, "dirichlet_alpha1p0": 1, "dirichlet_alpha0p5": 2,
                     "dirichlet_alpha0p1": 3}
            names = [c for c in ("auc_V_mean", "auc_V_rank_mean", "auc_V_late_mean")
                     if c in axis_runs]
            wide = axis_runs.pivot_table(index=keys, columns="axis", values=names,
                                         aggfunc="first")
            partitions = sorted(set(wide.index.get_level_values("partition_id")),
                                key=lambda p: order.get(p, 99))
            fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(partitions)), 4.2))
            width = 0.8 / len(names)
            x = np.arange(len(partitions))
            for k, name in enumerate(names):
                gap = wide[(name, "llm")] - wide[(name, "llm_random")]
                by_part = gap.groupby(level="partition_id")
                ax.bar(x + (k - (len(names) - 1) / 2) * width,
                       by_part.mean().reindex(partitions), width,
                       yerr=by_part.std().reindex(partitions).fillna(0), capsize=2,
                       color=GROUP_COLORS[k], label=name[4:])
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(partitions, fontsize=8)
            ax.set_ylabel("AUC(llm) - AUC(llm_random)")
            ax.set_title("H2: gain of the LLM units over size-matched random units")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            paths.append(_save(fig, folder, "llm_gain_by_partition.png"))
    return paths
