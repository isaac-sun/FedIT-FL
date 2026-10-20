# How the LLM units were made / LLM 单元是怎么来的

**English.** The labels were written once, before any experiment, and are frozen (`reference_llm_units.csv`; `main.py` stores its SHA-256 in every checkpoint, so a changed file never mixes with old results).

1. Input. For each of the 480 reference rows (60 per Dolly category, fixed split `split_seed=0`), one line: `row_id|category|instruction [C: first ~90 characters of the context] [R: first ~60 characters of the response]`. Example:
   `385|closed|Where did Gretel attend primary school? [C: Gretel Beer was born in Vienna into a Jewish family. She was mostly raised by her aunt Olg]`
2. Labeller. Claude (Anthropic) did the grouping itself in the project conversation of 2026-10-20, with no separate API call: it read all 480 lines, defined about ten units by **topic** (what the text is about, not the task type, which the Dolly categories already give), aiming for at least about 30 rows per unit, wrote ordered tie-break rules, then assigned every row to exactly one unit. The unit definitions and rules are in `units.json`.
3. The rows were labelled in four batches of 120. After the first full pass the unit "stories, books, film, TV, games, music and arts" had 89 rows, so it was split into `stories_media` (56) and `music_arts` (33) and the rows of that unit were relabelled; nothing else changed.
4. Only reference rows are labelled. Training rows never need a unit: the per-group Shapley value of a client is measured on the server's reference set.

A different LLM or prompt can be used: write a CSV with the columns `row_id, unit_id` for the same 480 rows and pass it with `--llm-units-file`. Its results go next to the old ones only if the file differs (checkpoint identity).

**中文。** 标签在任何实验之前一次性写好并冻结（`reference_llm_units.csv`；`main.py` 把它的 SHA-256 写进每个断点，换了文件不会和旧结果混在一起）。

1. 输入：480 条参考样本（每个 Dolly 类别 60 条，固定划分 `split_seed=0`）每条一行：`row_id|类别|指令 [C: 上下文前约 90 个字符] [R: 回答前约 60 个字符]`。
2. 标注者：Claude（Anthropic）在 2026-10-20 的项目对话里直接完成，没有单独调用 API。它读完 480 行，按**主题**（文本讲的是什么，而不是 Dolly 类别已经给出的任务类型）定义了约 10 个单元，每个单元目标至少约 30 条，写下有序的判定规则，再把每条样本分到唯一一个单元。单元定义和规则见 `units.json`。
3. 分四批、每批 120 条标注。第一遍后"故事、书、影视、游戏、音乐和艺术"单元有 89 条，于是拆成 `stories_media`（56）和 `music_arts`（33），只重标了这个单元的样本。
4. 只标注参考集。训练样本不需要单元：客户端的分组 Shapley 值是在服务器的参考集上算的。

换别的 LLM 或提示词也可以：为同样 480 条写一个含 `row_id, unit_id` 两列的 CSV，用 `--llm-units-file` 传入。
