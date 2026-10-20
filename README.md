# FedIT-FL: CSDD free-rider detection on federated instruction tuning

[English](#english) · [中文](#中文)

---

## English

This project carries the CSDD detector (dispersion of per-class Shapley values) from 20NEWS-FL and ECL-FL to **federated instruction tuning of a small LLM with LoRA**. It is a standalone project: the attack implementations, the FedAvg aggregation, the partition functions, the Shapley sampler and the result layout are copied from ECL-FL / 20NEWS-FL; everything task-specific is new. The design is in the project document "FedIT-FL 联邦指令微调实验设计（小LLM + LoRA）".

### Design (defaults)

| # | Choice | Default used here | Alternatives in the design doc |
|---|---|---|---|
| 1 | Main dataset | Dolly-15k (original release, 15,014 records, 8 native categories, CC BY-SA 3.0; `data/README.md`) | Alpaca-GPT4; Fed-Aya |
| 2 | Second dataset | none yet; a small Alpaca-GPT4 matrix after the formal results | none |
| 3 | Model | Qwen2.5-0.5B **base** (Apache-2.0), frozen in bf16, LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down, 8.8M trainable parameters | Instruct version; SmolLM2-360M; Qwen3-0.6B |
| 4 | Non-IID axis | Dirichlet over 8 k-means clusters of TF-IDF vectors of the training texts (see below) | Dolly native categories |
| 5 | Local training | 10 optimizer steps per round, batch 16 (micro-batch 8 × 2), AdamW 1e-4 constant, fresh optimizer every round, gradient clipping 1.0, loss on response tokens only, max length 512 (response ≤ 256) | 1 epoch per round |
| 6 | Shapley permutations | 20 (as ECL and the paper) | 10 |
| 7 | Seeds | 3 (120 runs) | 5 (200 runs, as ECL) |
| 8 | Quality metric | test cross-entropy of response tokens and perplexity, overall and per category | + ROUGE-L in the last round |
| 9 | Repository | private `isaac-sun/FedIT-FL` | subfolder of ECL-FL |

Change from the design doc: the Dirichlet partition clusters are TF-IDF + LSA (100 dimensions) + k-means instead of base-model embeddings. They are deterministic, need no GPU at data preparation and no second pass over the model, and they are still independent of the compared groupings: ARI 0.077 against the native categories over the training set. They agree somewhat with the LLM units (ARI 0.211 on the reference set), because both are topical.

### Mapping from ECL-FL

| Item | ECL-FL | FedIT-FL |
|---|---|---|
| Data | ECL, 321 series | Dolly-15k: fixed splits (`split_seed` 0) into a **reference set** of 480 rows (60 per category), a **global hold-out** of 480 rows at natural proportions, a **test set** of 1,000 rows and 13,054 **training** rows for 20 clients (about 650 each) |
| Model | PatchTST, all 0.92M parameters communicated | Qwen2.5-0.5B frozen + LoRA; only the 8.8M LoRA tensors (fp32) are trained, uploaded, averaged and attacked |
| Loss | MSE | per-example mean cross-entropy of the response tokens and EOS (Alpaca-style prompt); group loss = mean over the group's examples |
| FL protocol | 20 clients, all every round, 25 rounds, sample-weighted FedAvg | **same**; LoRA A and B are averaged separately (FedIT), and the Shapley coalitions use the same rule |
| Partitions | IID, Dirichlet α 0.1 / 0.5 / 1.0 over k-means load groups | IID, Dirichlet α 0.1 / 0.5 / 1.0 over TF-IDF clusters; at least 100 examples per client |
| Attacks | DFR / SDFR / AFR at 10 / 30 / 50%, nested attacker sets, 2 warm-up rounds of DFR for SDFR / AFR | **same**, applied to the LoRA state |
| Shapley | 20 permutations, equal-weight coalitions (paper Eq. 14), shared across axes | **same**; one forward pass per coalition gives every evaluation example's loss, every grouping is an average of the same losses, so extra class axes are free |
| Scores | V, P, SV; AUC with lower = more suspicious | **same**; pre-registered before any run: `V_mean`, `V_rank_mean`, `V_late_mean` (rounds ≥ 3) |

### Shapley axes (all scored on the same client updates)

| Axis | "Classes" | Role |
|---|---|---|
| `llm` | 10 LLM knowledge units of the reference rows (`groups/llm_units/`) | **primary** |
| `llm_random` | random relabelling with the LLM unit sizes | core control for the meaning of the units |
| `category` | the 8 Dolly categories (task types) | conventional grouping |
| `random` | random relabelling with the category sizes | control for `category` |
| `cluster` | nearest TF-IDF partition cluster | the data-skew axis itself (shows what full alignment with the non-IID direction gives) |
| `global` | none: one group, the 480 global hold-out rows at natural proportions | global Shapley control; only SV is defined |

The SV of every class axis is the plain Shapley value of the loss on the 480 reference rows (the first layer of the classless control); `global` is the second layer, on its own natural-proportion rows. The `Axis Tests` sheet compares, paired over attack runs (condition × seed), the LLM units against every control (H1) and each class axis's V against both classless SVs; `llm_gap_trend` tests H2, that AUC(llm) − AUC(llm_random) grows as α shrinks (Spearman over runs against the non-IID strength).

LLM units: 10 topical units, 33–65 reference rows each (nature/science 65, technology 45, health/food 45, places/travel 54, stories/media 56, sports/outdoors 51, history/society 51, business/work 35, everyday life 45, music/arts 33). They were written once by Claude from each row's instruction and excerpts of its context and response, and frozen; only the server's reference rows are labelled, client data never goes to the LLM. Details and the rules: `groups/llm_units/README.md` and `PROMPT.md`. `main.py` stores the file's SHA-256 in every checkpoint.

### Running

Requires Python 3.10+ and an installed PyTorch (the platform's CUDA build is kept). `run.sh` installs transformers (≥4.45, <5), safetensors, numpy, pandas, scikit-learn, scipy, openpyxl and matplotlib into `pip_pkgs/`. The Dolly file ships with the code; the base model does not (about 1 GB).

```bash
bash start.sh fetch-model   # once: Qwen2.5-0.5B into model_cache/ (ModelScope, then hf-mirror, then huggingface.co; resumes)
bash start.sh lr-probe      # no-attack runs at LRS (default 5e-5 1e-4 2e-4 5e-4) in parallel, no Shapley
python3 lr_probe_report.py  # test loss by round per rate, round-0 share of the loss drop, a suggested rate
bash start.sh pilot         # 8 runs: seed 42, IID and α=0.1 × (no attack + DFR / SDFR / AFR 30%)
bash start.sh full          # 120 runs: 4 partitions × 10 conditions × 3 seeds, partition after partition
bash start.sh status        # running or not + latest log lines (each round logs its time: total=…s)
bash start.sh stop          # stop; run the same start command again to resume
```

In mainland China add `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`. If the model download is too slow on the server, download the seven files listed in `fetch_model.sh` on another computer into `model_cache/Qwen2.5-0.5B/`.

Environment overrides: `LOCAL_LR` (default 1e-4; any other rate adds `_lr<rate>` to the results folder), `GLOBAL_SIZE` (rows of the global axis, default 480; 0 = no global axis; other values add `_g<size>`; the training and test rows never change with it), `SEEDS` (e.g. `42,43,44,45,46`), `TASKS` (e.g. `"iid dirichlet:0.1"` to run only some partitions, also to share the matrix between two machines), `SHAPLEY_AXES` (first = primary), `EVAL_TOKEN_BUDGET` (padded tokens per evaluation batch, default 32768), `MODEL_PATH`, `FORCE_DEVICE`, `FOREGROUND=1`. `full-parallel` runs the four partitions as four processes on one GPU (about 8–10 GB of GPU memory each, inferred); because Shapley evaluation already keeps the GPU busy, it helps less than in ECL. A results folder refuses a run with other settings (learning rate, axes, global size, model, …), so finished runs are never mixed.

**Jupyter.** `run_fedit.ipynb` covers the whole workflow without a terminal: unzip, model download, environment check, learning-rate probe and its report, pilot, progress, results (tables and charts), formal matrix, per-run charts, packing results, stop. Every cell enters the project folder itself.

### Resume and outputs

After every round the LoRA global state, the last three global states, the attack estimates, the running statistics, all RNG states and each client's DataLoader RNG state are written atomically to `runs/<experiment>_seed<seed>/checkpoint.pt` (about 140 MB). Rerunning the same command skips runs with a `summary.json` and continues an unfinished run from its next round; the result equals an uninterrupted run (tested). A run whose loss or parameters become non-finite stops there, is kept with `diverged` in `summary.json`, and is left out of the AUC tables (`Diverged Runs`).

Per finished run (`runs/*/`): `experiment_results.xlsx` with the 20NEWS sheet layout (`round_shapley_details`, `experiment_summary`, `per_class_records`, `per_class_client_metrics`, `per_class_global_metrics`, `client_totals`, `axis_shapley_records`, `axis_client_totals`, `client_data_distribution`, `experiment_config`), the same tables as CSV, `summary.json` (AUCs per score and axis as `auc_<score>__<axis>`) and 7 charts. "Class" means the primary axis (LLM unit); accuracy columns become loss: `test_loss`, `test_ppl`, `test_group_<category>_loss`, `ref_group_<unit>_loss`, `class_<unit>_loss` of each client's submitted model, `local_train_loss`, `local_ref_loss`. The summary also has `untrained_test_loss`, `round0_share_of_loss_drop` (how much of the total loss drop round 0 takes) and `round0_v_scale` (honest V in round 0 over its mean from round 3).

Combined (`<results root>/fedit_csdd_summary.xlsx`, rebuilt after every run): AUC by Condition / by Run, Seed Macro, Client Totals, Test Loss, Round Curves, Completeness, Diverged Runs, Robustness AUC (post-hoc rank, late and log aggregations), Axis Comparison, Axis AUC by Condition / by Run, Axis Definitions, **Axis Tests**, and all detail tables concatenated. Charts in `plots/`: `auc_heatmap.png`, `auc_by_condition.png`, `final_test_loss.png`, `auc_v_robustness.png`, `auc_v_p_sv.png`, `auc_by_axis.png`, `llm_gain_by_partition.png` (H2).

### Cost (inferred, not measured)

Shapley dominates. Per round about 365 distinct coalitions are evaluated, each one forward pass over the reference and global rows (about 185k tokens, estimated from word counts), plus 20 client evaluations, training and the test set. From FLOPs this is roughly 10–16 minutes per round on one RTX 5090, 4–7 hours per run: the pilot about 1.5–2.5 days and the 120-run matrix about 3–5 weeks on one GPU. The global axis is about 45% of the Shapley time. The first pilot round logs the real time (`bash start.sh status`, `total=…s`); the plan can be resized from it (two machines with `TASKS`, `GLOBAL_SIZE`, 10 permutations).

### Tests

```bash
python3 -m pytest -q tests
```

Uses a tiny random Qwen2 model and a BPE tokenizer trained on the fly (no download, CPU). Covers the data file and fixed splits, the frozen LLM units, prompt encoding and truncation, LoRA (starts at the base model, state round trip, FedAvg), the efficiency property of the per-group Shapley values, the AUC, the axis tests, an end-to-end run of all four conditions with all six axes, resume after an interruption (identical result), skipping finished runs, the settings guard and the learning-rate probe report.

---

## 中文

本项目把 CSDD 检测（逐类 Shapley 值的离散度）从 20NEWS-FL 和 ECL-FL 迁移到**小 LLM + LoRA 的联邦指令微调**。它是独立项目：攻击、FedAvg、划分函数、Shapley 采样和结果表格式取自 ECL-FL / 20NEWS-FL，任务相关部分是新写的。设计见项目文档《FedIT-FL 联邦指令微调实验设计（小LLM + LoRA）》。

### 设计（默认选择）

| # | 选择 | 本代码采用 | 设计文档里的备选 |
|---|---|---|---|
| 1 | 主数据集 | Dolly-15k（原始发布版 15,014 条，8 个原生类别，CC BY-SA 3.0，见 `data/README.md`） | Alpaca-GPT4；Fed-Aya |
| 2 | 第二数据集 | 暂不加；正式结果出来后再跑 Alpaca-GPT4 小矩阵 | 不加 |
| 3 | 模型 | Qwen2.5-0.5B **基座版**（Apache-2.0），bf16 冻结；LoRA r=16、alpha=32、dropout 0.05，作用于 q/k/v/o/gate/up/down，可训练 8.8M 参数 | Instruct 版；SmolLM2-360M；Qwen3-0.6B |
| 4 | 非 IID 划分轴 | 训练文本 TF-IDF 向量的 k-means 8 簇上做 Dirichlet（见下） | Dolly 原生类别 |
| 5 | 本地训练 | 每轮 10 步，batch 16（微批 8 × 2），AdamW 1e-4 恒定，每轮重置优化器，梯度裁剪 1.0，只对回复 token 算损失，最大长度 512（回复 ≤ 256） | 每轮 1 个 epoch |
| 6 | Shapley 排列数 | 20（同 ECL 和论文） | 10 |
| 7 | 种子数 | 3（120 个运行） | 5（200 个，同 ECL） |
| 8 | 质量指标 | 测试集回复 token 交叉熵和困惑度，整体和按类别 | 最后一轮加 ROUGE-L |
| 9 | 代码仓库 | 私有仓库 `isaac-sun/FedIT-FL` | ECL-FL 子目录 |

与设计文档的不同：Dirichlet 用的聚类改成 TF-IDF + LSA（100 维）+ k-means，而不是基座模型嵌入。它是确定性的，数据准备不需要 GPU，也不用先把模型过一遍，并且仍然独立于被比较的分组：在训练集上与原生类别的 ARI 为 0.077。它和 LLM 单元有一定一致性（参考集上 ARI 0.211），因为两者都按主题。

### 与 ECL-FL 的对应

| 项 | ECL-FL | FedIT-FL |
|---|---|---|
| 数据 | ECL 321 条序列 | Dolly-15k，固定划分（`split_seed` 0）：**参考集** 480 条（每类 60），**全局留出集** 480 条（自然比例），**测试集** 1,000 条，其余 13,054 条**训练**数据分给 20 个客户端（每个约 650 条） |
| 模型 | PatchTST，0.92M 参数全部通信 | Qwen2.5-0.5B 冻结 + LoRA；只有 8.8M 个 LoRA 参数（fp32）参与训练、上传、聚合和攻击 |
| 损失 | MSE | 每条样本回复 token（含 EOS）的平均交叉熵（Alpaca 式提示词）；组损失 = 组内样本平均 |
| FL 协议 | 20 客户端全参与，25 轮，样本量加权 FedAvg | **相同**；LoRA 的 A、B 分别平均（FedIT 做法），Shapley 联盟用同样的规则 |
| 划分 | IID，按 k-means 负荷组做 Dirichlet α 0.1 / 0.5 / 1.0 | IID，按 TF-IDF 聚类做 Dirichlet α 0.1 / 0.5 / 1.0；每客户端至少 100 条 |
| 攻击 | DFR / SDFR / AFR，10 / 30 / 50%，嵌套攻击者集合，SDFR / AFR 前 2 轮用 DFR | **相同**，作用在 LoRA 状态上 |
| Shapley | 20 个排列，等权联盟（论文式 14），各轴共用 | **相同**；每个联盟一次前向得到每条评估样本的损失，各分组只是对同一批损失求不同的平均，所以多加类别轴不增加时间 |
| 指标 | V、P、SV；AUC 越低越可疑 | **相同**；运行前预注册 `V_mean`、`V_rank_mean`、`V_late_mean`（第 3 轮起）为主指标 |

### Shapley 分组轴（全部基于同一批客户端更新）

| 轴 | "类别" | 角色 |
|---|---|---|
| `llm` | 参考集的 10 个 LLM 知识单元（`groups/llm_units/`） | **主组** |
| `llm_random` | 与 LLM 单元大小相同的随机重标 | 单元语义的核心对照 |
| `category` | Dolly 8 个原生类别（任务类型） | 常规分组 |
| `random` | 与原生类别大小相同的随机重标 | `category` 的对照 |
| `cluster` | 最近的 TF-IDF 划分簇 | 数据偏斜方向本身（表示与非 IID 方向完全对齐时的效果） |
| `global` | 不分类别：一个组，480 条自然比例的全局留出样本 | 全局 Shapley 对照，只有 SV |

每个类别轴的 SV 就是 480 条参考样本上整体损失的普通 Shapley 值（不分类对照的第一层）；`global` 是第二层，用另外一批自然比例样本。`Axis Tests` 表按攻击运行（条件 × 种子）配对比较：LLM 单元对每个对照（H1），以及每个类别轴的 V 对两层不分类 SV；`llm_gap_trend` 检验 H2，即 AUC(llm) − AUC(llm_random) 是否随 α 变小而增大（各运行与非 IID 强度的 Spearman 相关）。

LLM 单元：10 个主题单元，每个 33–65 条参考样本（自然与科学 65、技术 45、健康饮食 45、地点旅行 54、故事媒体 56、体育户外 51、历史社会 51、商业工作 35、日常生活 45、音乐艺术 33）。由 Claude 根据每条样本的指令和上下文、回答的片段一次性写定并冻结；只标注服务器参考集，客户端数据不交给 LLM。做法和规则见 `groups/llm_units/README.md` 与 `PROMPT.md`。`main.py` 把该文件的 SHA-256 写进每个断点。

### 运行

需要 Python 3.10+ 和已安装的 PyTorch（保留平台自带的 CUDA 版本）。`run.sh` 把 transformers（≥4.45，<5）、safetensors、numpy、pandas、scikit-learn、scipy、openpyxl、matplotlib 装进 `pip_pkgs/`。Dolly 数据随代码提供；基座模型（约 1 GB）不在包里。

```bash
bash start.sh fetch-model   # 只需一次：下载 Qwen2.5-0.5B 到 model_cache/（依次 ModelScope、hf-mirror、huggingface.co，可续传）
bash start.sh lr-probe      # 无攻击、不算 Shapley，同时跑 LRS 里的学习率（默认 5e-5 1e-4 2e-4 5e-4）
python3 lr_probe_report.py  # 每个学习率各轮的测试损失、第 0 轮占总下降的比例、建议学习率
bash start.sh pilot         # 试点 8 个：seed 42，IID 和 α=0.1 ×（无攻击 + DFR / SDFR / AFR 30%）
bash start.sh full          # 正式 120 个：4 种划分 × 10 个条件 × 3 个种子，按划分依次运行
bash start.sh status        # 是否在运行 + 最新日志（每轮都记录耗时 total=…s）
bash start.sh stop          # 停止；再运行同一条启动命令即可续跑
```

国内服务器加 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。服务器下载模型太慢时，在别的电脑上下载 `fetch_model.sh` 里列出的 7 个文件，放进 `model_cache/Qwen2.5-0.5B/`。

环境变量：`LOCAL_LR`（默认 1e-4，其他值会在结果文件夹名后加 `_lr<值>`）、`GLOBAL_SIZE`（全局轴样本数，默认 480，0 表示不要全局轴，其他值加 `_g<值>`；训练集和测试集不随它改变）、`SEEDS`（如 `42,43,44,45,46`）、`TASKS`（如 `"iid dirichlet:0.1"`，只跑部分划分，也可用来在两台机器上分摊）、`SHAPLEY_AXES`（第一个为主组）、`EVAL_TOKEN_BUDGET`（每个评估批次的 token 数，默认 32768）、`MODEL_PATH`、`FORCE_DEVICE`、`FOREGROUND=1`。`full-parallel` 在一张卡上同时跑 4 个划分（每个约 8–10 GB 显存，推算）；Shapley 评估本身已经让 GPU 很忙，提速不如 ECL 明显。结果文件夹会拒绝设置不同的运行（学习率、分组轴、全局轴大小、模型等），已完成的结果不会被混用。

**Jupyter。** `run_fedit.ipynb` 不用终端就能走完全流程：解压、下载模型、环境检查、学习率探测及报告、试点、进度、结果（表和图）、正式实验、单个实验的图、打包、停止。每一格都会先进入项目目录。

### 断点续跑和输出

每轮结束后把 LoRA 全局状态、前三轮全局状态、攻击估计量、累计统计、全部随机数状态和每个客户端 DataLoader 的随机数状态原子写入 `runs/<实验>_seed<种子>/checkpoint.pt`（约 140 MB）。再次运行同一条命令时，有 `summary.json` 的小实验会跳过，未完成的从下一轮继续；结果与不中断时完全一致（已测试）。损失或参数出现非有限值时该运行在此结束，`summary.json` 里标 `diverged`，不进 AUC 表（列在 `Diverged Runs`）。

每个完成的小实验（`runs/*/`）：20NEWS 格式的 `experiment_results.xlsx`（`round_shapley_details`、`experiment_summary`、`per_class_records`、`per_class_client_metrics`、`per_class_global_metrics`、`client_totals`、`axis_shapley_records`、`axis_client_totals`、`client_data_distribution`、`experiment_config`），同样内容的 CSV，`summary.json`（各指标、各轴的 AUC，键名 `auc_<指标>__<轴>`）和 7 张图。"类别"指主组（LLM 单元）；准确率列换成损失：`test_loss`、`test_ppl`、`test_group_<类别>_loss`、`ref_group_<单元>_loss`、每个客户端提交模型的 `class_<单元>_loss`、`local_train_loss`、`local_ref_loss`。汇总里还有 `untrained_test_loss`、`round0_share_of_loss_drop`（第 0 轮占总损失下降的比例）和 `round0_v_scale`（诚实客户端第 0 轮 V 与第 3 轮起平均 V 之比）。

总表（`<结果目录>/fedit_csdd_summary.xlsx`，每完成一个运行就重建）：AUC by Condition / by Run、Seed Macro、Client Totals、Test Loss、Round Curves、Completeness、Diverged Runs、Robustness AUC（事后的排名、后期、对数聚合）、Axis Comparison、Axis AUC by Condition / by Run、Axis Definitions、**Axis Tests**，以及所有明细表的合并。`plots/` 里的总图：`auc_heatmap.png`、`auc_by_condition.png`、`final_test_loss.png`、`auc_v_robustness.png`、`auc_v_p_sv.png`、`auc_by_axis.png`、`llm_gain_by_partition.png`（H2）。

### 耗时（推算，未实测）

Shapley 是大头。每轮约 365 个不同联盟，每个对参考集和全局样本做一次前向（约 18.5 万 token，按词数估计），另有 20 次客户端评估、训练和测试集评估。按 FLOPs 推算，一张 RTX 5090 上每轮约 10–16 分钟，每个运行 4–7 小时：试点约 1.5–2.5 天，120 个正式运行在一张卡上约 3–5 周。全局轴约占 Shapley 时间的 45%。试点第一轮会记录实际耗时（`bash start.sh status` 里的 `total=…s`），可据此调整计划（用 `TASKS` 分两台机器、`GLOBAL_SIZE`、10 个排列）。

### 测试

```bash
python3 -m pytest -q tests
```

用一个随机初始化的小 Qwen2 模型和现场训练的 BPE 分词器（不下载，CPU）。覆盖数据文件和固定划分、冻结的 LLM 单元、提示词编码和截断、LoRA（初始等于基座模型、状态往返、FedAvg）、逐组 Shapley 的有效性（各客户端 SV 之和等于全联盟的损失下降）、AUC、轴间检验、四个条件 × 六个分组轴的端到端运行、中断后续跑结果一致、跳过已完成运行、设置不同时拒绝复用结果文件夹，以及学习率探测报告。
