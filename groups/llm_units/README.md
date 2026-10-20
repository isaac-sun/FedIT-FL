# Dolly LLM knowledge units / Dolly 的 LLM 知识单元

**English.** Frozen LLM grouping of the 480 Shapley reference rows into 10 topical knowledge units: the primary CSDD grouping axis (`llm`). Files: `reference_llm_units.csv` (row → unit, the file experiments read; the instruction column is only for reading), `units.json` (unit definitions and ordered rules), `PROMPT.md` (what the LLM saw and was asked), `check_units.py` (checks coverage and sizes, writes the cross-tables), `crosstab_vs_category.csv`, `crosstab_vs_cluster.csv`.

Agreement with the native Dolly categories is near zero (ARI 0.029): the units are about topic, the categories about task type, so the two class axes measure different things. Agreement with the TF-IDF clusters that drive the Dirichlet partition is ARI 0.211: both are topical, so under non-IID the LLM units partly follow the direction of the data skew. The `cluster` axis (the partition clusters themselves) shows what full alignment gives, and `llm_random` (random units of the same sizes) is the control for the semantic gain.

**中文。** 把 480 条 Shapley 参考样本冻结地分成 10 个主题型知识单元，作为 CSDD 的主分组轴（`llm`）。实验读取 `reference_llm_units.csv`（指令一列只供阅读）；`units.json` 是单元定义和有序规则；`PROMPT.md` 记录 LLM 看到了什么、被要求做什么；`check_units.py` 检查覆盖和大小并写交叉表。

与 Dolly 原生类别几乎无关（ARI 0.029）：单元按主题、类别按任务类型，两个类别轴衡量的是不同的东西。与驱动 Dirichlet 划分的 TF-IDF 聚类 ARI 为 0.211：两者都是主题型的，所以非 IID 下 LLM 单元部分顺着数据偏斜的方向。`cluster` 轴（划分所用的聚类本身）表示完全对齐时的效果，`llm_random`（大小相同的随机单元）是语义增益的对照。

| id | unit / 单元 | rows |
|---|---|---|
| 0 | nature_science / 自然与科学（含数学） | 65 |
| 1 | technology / 技术、工程、计算机、交通工具 | 45 |
| 2 | health_food / 健康、饮食、健身 | 45 |
| 3 | places_travel / 地点、地理、旅行 | 54 |
| 4 | stories_media / 故事、书籍、影视、游戏 | 56 |
| 5 | sports_outdoors / 体育与户外（含大学吉祥物） | 51 |
| 6 | history_society / 历史、政治、社会、宗教、哲学、人物 | 51 |
| 7 | business_work / 商业、金钱、工作、职业 | 35 |
| 8 | everyday_life / 日常生活、家庭、人际、常识与文字谜题 | 45 |
| 9 | music_arts / 音乐与视觉、表演艺术 | 33 |
