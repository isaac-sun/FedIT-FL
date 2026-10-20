from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for one federated instruction-tuning run (FedIT-FL).

    The FL protocol (clients, participation, rounds, sample-weighted FedAvg,
    attacks with nested attacker sets, equal-weight Shapley coalitions with
    shared permutations) is the one of ECL-FL and the 20NEWS-FL formal
    matrix.  The task-specific parts are the data (Dolly-15k), the model
    (a small base LLM with LoRA), the loss (response-token cross-entropy) and
    the "classes" behind the per-group Shapley values.
    """

    # ── Data (Dolly-15k, data/dolly.py) ─────────────────────────────────
    data_path: str = "data/databricks-dolly-15k.jsonl"
    check_data_sha256: bool = True
    split_seed: int = 0                  # fixed splits, independent of the run seed
    reference_per_category: int = 60     # Shapley reference set: 8 x 60 = 480 rows
    # "global" Shapley axis: this many rows at natural proportions, one group
    # (0 = no global axis; it costs about as much Shapley time as the
    # reference set, see README).
    global_size: int = 480
    global_holdout: int = 480            # rows held out for it, whatever global_size is
    test_size: int = 1000
    max_length: int = 512                # prompt + response tokens
    max_response_length: int = 256
    # Partition axis for Dirichlet non-IID: k-means clusters of TF-IDF vectors
    # of the training texts, independent of every compared grouping.
    partition_clusters: int = 8
    cluster_seed: int = 0

    # ── Partition ───────────────────────────────────────────────────────
    num_clients: int = 20
    partition_method: str = "iid"        # "iid" | "dirichlet"
    dirichlet_alpha: float | None = None
    partition_id: str = "iid"
    partition_seed: int = 42
    min_samples_per_client: int = 100

    # ── Model ───────────────────────────────────────────────────────────
    model_path: str = "model_cache/Qwen2.5-0.5B"
    bf16: bool = True                    # base weights in bf16 on CUDA
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj")
    lora_init_seed: int = 0              # same LoRA A init for every run and seed
    lm_head_chunk: int = 2048            # response tokens per LM-head chunk

    # ── Federated learning ──────────────────────────────────────────────
    num_rounds: int = 25
    participation_ratio: float = 1.0     # 20 of 20 clients every round
    local_steps: int = 10                # optimizer steps per client and round
    batch_size: int = 16                 # examples per optimizer step
    micro_batch_size: int = 8            # examples per forward/backward pass
    local_lr: float = 1e-4
    local_lr_schedule: str = "constant"  # "constant" | "linear_decay" (within a round)
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    eval_token_budget: int = 32768       # padded tokens per evaluation batch
    num_workers: int = 0

    # ── Shapley (CSDD) ──────────────────────────────────────────────────
    compute_shapley: bool = True
    num_mc_samples: int = 20
    # Groupings of the reference rows (see main.AXIS_CHOICES); the first is
    # the primary axis behind the 20NEWS-style per-run tables.  All axes come
    # from the same per-example losses, so extra class axes cost nothing;
    # only "global" adds evaluation work (its own rows).
    shapley_axes: tuple = ("llm", "llm_random", "category", "random", "cluster", "global")
    llm_units_file: str = "groups/llm_units/reference_llm_units.csv"
    llm_units_sha256: str = ""
    group_seed: int = 0
    # Equal-weight coalitions, as in the paper (Eq. 14).
    shapley_equal_weights: bool = True
    shapley_cache_coalitions: bool = True

    # ── Attacks (copied from 20NEWS-FL / ECL-FL, unchanged) ─────────────
    attack_type: str = "none"            # "none" | "dfr" | "sdfr" | "afr"
    malicious_ratio: float = 0.1
    attack_warmup_rounds: int = 2        # SDFR and AFR send DFR noise first
    dfr_gamma: float = 1.0
    dfr_sigma_override: float | None = None
    afr_c_override: float | None = None
    afr_noisy_frac: float = 0.1

    # ── Runtime ─────────────────────────────────────────────────────────
    seed: int = 42
    experiment_name: str = "baseline"
    results_dir: str = "results"
    device: str = "cpu"

    @property
    def shapley_axis(self) -> str:
        return self.shapley_axes[0]
