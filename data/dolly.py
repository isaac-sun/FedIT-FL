"""Dolly-15k for federated instruction tuning.

The file is the original Databricks release (``data/databricks-dolly-15k.jsonl``,
15,014 records, CC BY-SA 3.0, see ``data/README.md``).  Its SHA-256 is checked
so that the fixed splits, and the LLM knowledge units written for the rows of
the reference set, stay valid.

Fixed splits (``split_seed``, independent of the run seed):

* ``reference``: the server's Shapley reference set, ``reference_per_category``
  rows of each of the 8 native categories (balanced, as the 20NEWS and ECL
  reference sets);
* ``global``: ``global_size`` rows at natural proportions, for the classless
  "global" Shapley axis (taken from a fixed hold-out of ``global_holdout``
  rows, so the other splits do not depend on ``global_size``);
* ``test``: ``test_size`` rows at natural proportions, for model quality;
* ``train``: everything else, split over the clients.

The partition axis for Dirichlet non-IID is a k-means clustering of TF-IDF
vectors of the training texts (``partition_clusters`` clusters).  None of the
compared Shapley groupings is built from it, so no grouping gets the "home
advantage" the k-means load groups had in ECL-FL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CATEGORIES = ("brainstorming", "classification", "closed_qa", "creative_writing",
              "general_qa", "information_extraction", "open_qa", "summarization")
DOLLY_SHA256 = "e442d96b320010510c67a6cc0a86ea988a2a6f71c7570d1afbdf4fa631896e58"

# Alpaca-style prompt for a base (not instruction-tuned) model.
PROMPT_HEAD = "Below is an instruction that describes a task. Write a response that " \
              "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
PROMPT_CONTEXT = "### Input:\n{context}\n\n"
PROMPT_TAIL = "### Response:\n"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: str, check_sha256: bool = True) -> pd.DataFrame:
    """All Dolly rows with ``row_id`` = line number (0-based)."""
    if check_sha256:
        actual = file_sha256(path)
        if actual != DOLLY_SHA256:
            raise SystemExit(
                f"{path}: SHA-256 {actual} differs from the expected {DOLLY_SHA256}; "
                "the fixed splits and the LLM unit labels belong to the original file")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame.insert(0, "row_id", np.arange(len(frame)))
    frame["category_id"] = frame["category"].map({c: i for i, c in enumerate(CATEGORIES)})
    if frame["category_id"].isna().any():
        raise ValueError("unknown Dolly category in the data file")
    frame["category_id"] = frame["category_id"].astype(int)
    for column in ("instruction", "context", "response"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def make_splits(frame: pd.DataFrame, reference_per_category: int, global_size: int,
                test_size: int, seed: int, global_holdout: int = 480) -> dict:
    """Disjoint row-id arrays: reference (balanced), global, test, train.

    ``max(global_holdout, global_size)`` rows are held out for the global
    axis whatever ``global_size`` is (the global rows are the first
    ``global_size`` of them), so a smaller or no global axis leaves the test
    and training rows unchanged.
    """
    rng = np.random.default_rng(seed)
    reference = []
    for category in range(len(CATEGORIES)):
        rows = frame.loc[frame["category_id"] == category, "row_id"].to_numpy()
        reference.append(np.sort(rng.choice(rows, reference_per_category, replace=False)))
    reference = np.concatenate(reference)
    rest = rng.permutation(np.setdiff1d(frame["row_id"].to_numpy(), reference))
    holdout = max(global_holdout, global_size)
    global_rows = np.sort(rest[:global_size])
    test_rows = np.sort(rest[holdout:holdout + test_size])
    train_rows = np.sort(rest[holdout + test_size:])
    return {"reference": reference, "global": global_rows, "test": test_rows,
            "train": train_rows}


def text_clusters(train_texts: list[str], other_texts: list[str], num_clusters: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means clusters of TF-IDF vectors (100-dim LSA, unit length).

    Fitted on the training texts; ``other_texts`` (the reference rows) get the
    nearest cluster, for the "cluster" Shapley axis.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True,
                                 stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(train_texts)
    dims = max(2, min(100, matrix.shape[1] - 1, matrix.shape[0] - 1))
    svd = TruncatedSVD(dims, random_state=seed)

    def unit(vectors):
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    vectors = unit(svd.fit_transform(matrix))
    kmeans = KMeans(num_clusters, n_init=10, random_state=seed).fit(vectors)
    other = kmeans.predict(unit(svd.transform(vectorizer.transform(other_texts)))) \
        if other_texts else np.zeros(0)
    return kmeans.labels_.astype(np.int64), np.asarray(other, dtype=np.int64)


def full_text(frame: pd.DataFrame) -> list[str]:
    return (frame["instruction"] + " " + frame["context"] + " " + frame["response"]).tolist()


# ── tokenization ─────────────────────────────────────────────────────────────

def encode_rows(tokenizer, frame: pd.DataFrame, max_length: int,
                max_response_length: int) -> list[dict]:
    """Prompt + response token ids; the loss covers the response and EOS only.

    Long examples are cut in this order: the response to ``max_response_length``
    tokens (EOS included), then the context, then the instruction, so the
    "### Response:" marker always stays in front of the response.
    """
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("the tokenizer has no EOS token")

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"] if text else []

    encoded = []
    for row in frame.itertuples(index=False):
        response = ids(row.response)[:max(max_response_length - 1, 1)] + [eos]
        tail = ids(PROMPT_TAIL)
        head = ids(PROMPT_HEAD.format(instruction=row.instruction))
        context = ids(PROMPT_CONTEXT.format(context=row.context)) if row.context.strip() else []
        budget = max_length - len(response) - len(tail)
        if len(head) + len(context) > budget:
            context = context[:max(budget - len(head), 0)]
        if len(head) > budget:
            head = head[:max(budget, 1)]
        prompt = head + context + tail
        encoded.append({"input_ids": prompt + response, "prompt_length": len(prompt),
                        "row_id": int(row.row_id)})
    return encoded


class TokenDataset(Dataset):
    """Encoded examples with a group label (native category by default)."""

    def __init__(self, examples: list[dict], groups):
        self.examples = examples
        self.groups = np.asarray(groups, dtype=np.int64)
        if len(self.groups) != len(examples):
            raise ValueError("one group label per example is required")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return example["input_ids"], example["prompt_length"], int(self.groups[index])

    def lengths(self) -> np.ndarray:
        return np.array([len(e["input_ids"]) for e in self.examples])

    def row_ids(self) -> np.ndarray:
        return np.array([e["row_id"] for e in self.examples])


def make_collate(pad_token_id: int):
    """Right-padded ``input_ids``, ``attention_mask``, ``labels`` (-100 = no loss)."""
    def collate(items):
        width = max(len(ids) for ids, _, _ in items)
        input_ids = torch.full((len(items), width), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(items), width), dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        groups = torch.zeros(len(items), dtype=torch.long)
        for i, (ids, prompt_length, group) in enumerate(items):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention[i, :len(ids)] = 1
            labels[i, prompt_length:len(ids)] = torch.tensor(ids[prompt_length:])
            groups[i] = group
        return input_ids, attention, labels, groups
    return collate


# ── everything a run needs ───────────────────────────────────────────────────

@dataclass
class DollyData:
    frame: pd.DataFrame
    splits: dict
    train: TokenDataset
    reference: TokenDataset
    global_set: TokenDataset | None
    test: TokenDataset
    partition_clusters: np.ndarray          # one label per training example
    reference_clusters: np.ndarray          # nearest partition cluster of each reference row
    num_categories: int = len(CATEGORIES)
    categories: tuple = CATEGORIES
    cluster_sizes: list = field(default_factory=list)

    def train_categories(self) -> np.ndarray:
        return self.train.groups


def load_dolly(config, tokenizer) -> DollyData:
    frame = load_records(config.data_path, check_sha256=config.check_data_sha256)
    splits = make_splits(frame, config.reference_per_category, config.global_size,
                         config.test_size, config.split_seed, config.global_holdout)
    by_row = frame.set_index("row_id")

    def dataset(rows):
        part = by_row.loc[rows].reset_index()
        examples = encode_rows(tokenizer, part, config.max_length, config.max_response_length)
        return TokenDataset(examples, part["category_id"].to_numpy())

    train = dataset(splits["train"])
    clusters, reference_clusters = text_clusters(
        full_text(by_row.loc[splits["train"]].reset_index()),
        full_text(by_row.loc[splits["reference"]].reset_index()),
        config.partition_clusters, config.cluster_seed)
    return DollyData(
        frame=frame, splits=splits, train=train,
        reference=dataset(splits["reference"]),
        global_set=dataset(splits["global"]) if config.global_size > 0 else None,
        test=dataset(splits["test"]),
        partition_clusters=clusters,
        reference_clusters=reference_clusters,
        cluster_sizes=np.bincount(clusters, minlength=config.partition_clusters).tolist(),
    )


def load_reference_units(path: str, reference_rows: np.ndarray) -> np.ndarray:
    """Frozen LLM unit label of every reference row (CSV: row_id, unit_id)."""
    table = pd.read_csv(path)
    mapping = dict(zip(table["row_id"].astype(int), table["unit_id"].astype(int)))
    missing = [int(r) for r in reference_rows if int(r) not in mapping]
    if missing:
        raise SystemExit(f"{path}: no LLM unit for reference rows {missing[:10]} "
                         f"({len(missing)} missing); the units belong to other splits")
    units = np.array([mapping[int(r)] for r in reference_rows], dtype=np.int64)
    present = np.unique(units)
    return np.searchsorted(present, units)   # contiguous 0..K-1


def random_labels_like(labels: np.ndarray, seed: int) -> np.ndarray:
    """A random relabelling with exactly the same group sizes."""
    return np.random.default_rng(seed).permutation(np.asarray(labels))
