#!/usr/bin/env python3
"""Check the frozen LLM units and write their cross-tables.

Usage (from the project root):
    python3 groups/llm_units/check_units.py

Checks that ``reference_llm_units.csv`` covers exactly the Shapley reference
rows of the fixed split (data/dolly.py, split_seed 0), that every unit has at
least 20 rows, and that the unit names match ``units.json``.  Writes
``crosstab_vs_category.csv`` and ``crosstab_vs_cluster.csv`` (cluster = the
TF-IDF k-means clusters that drive the Dirichlet partition) and prints the
adjusted Rand index against both.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from config import Config  # noqa: E402
from data.dolly import full_text, load_records, make_splits, text_clusters  # noqa: E402


def main():
    from sklearn.metrics import adjusted_rand_score

    config = Config()
    frame = load_records(config.data_path)
    splits = make_splits(frame, config.reference_per_category, config.global_size,
                         config.test_size, config.split_seed, config.global_holdout)
    units = json.load(open(os.path.join(HERE, "units.json")))["units"]
    names = {u["unit_id"]: u["unit_name"] for u in units}
    table = pd.read_csv(os.path.join(HERE, "reference_llm_units.csv"))
    if sorted(table["row_id"]) != sorted(splits["reference"].tolist()):
        sys.exit("reference_llm_units.csv does not cover exactly the reference rows")
    bad = table[table["unit_name"] != table["unit_id"].map(names)]
    if len(bad):
        sys.exit(f"unit names differ from units.json in {len(bad)} rows")
    sizes = table["unit_id"].value_counts().sort_index()
    if sizes.min() < 20:
        sys.exit(f"unit(s) with fewer than 20 rows: {sizes[sizes < 20].to_dict()}")
    by_row = frame.set_index("row_id")
    _, clusters = text_clusters(full_text(by_row.loc[splits["train"]].reset_index()),
                                full_text(by_row.loc[table["row_id"]].reset_index()),
                                config.partition_clusters, config.cluster_seed)
    table["cluster"] = clusters
    unit_label = (table["unit_id"].astype(str) + "_" + table["unit_name"]).rename("unit")
    pd.crosstab(unit_label, table["category"]).to_csv(
        os.path.join(HERE, "crosstab_vs_category.csv"))
    pd.crosstab(unit_label, table["cluster"].rename("cluster")).to_csv(
        os.path.join(HERE, "crosstab_vs_cluster.csv"))
    print("rows per unit:", {names[k]: int(v) for k, v in sizes.items()})
    print(f"ARI vs Dolly category: {adjusted_rand_score(table['category'], table['unit_id']):.3f}")
    print(f"ARI vs TF-IDF cluster: {adjusted_rand_score(table['cluster'], table['unit_id']):.3f}")


if __name__ == "__main__":
    main()
