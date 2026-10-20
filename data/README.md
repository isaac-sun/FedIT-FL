# Data / 数据

`databricks-dolly-15k.jsonl` is the original release of **databricks-dolly-15k** by Databricks, Inc., taken from <https://github.com/databrickslabs/dolly> at commit `2305eb7f2f4b3beb2379f34c6addf335b46c4b43` (2023-04-13, "Removing record from dataset."), file `data/databricks-dolly-15k.jsonl`. 15,014 records; SHA-256 `e442d96b320010510c67a6cc0a86ea988a2a6f71c7570d1afbdf4fa631896e58` (checked by `data/dolly.py`).

License: **Creative Commons Attribution-ShareAlike 3.0 Unported** (CC BY-SA 3.0, <https://creativecommons.org/licenses/by-sa/3.0/>). Copyright Databricks, Inc. The file is redistributed here unchanged. Some records contain text from Wikipedia (also CC BY-SA 3.0).

The Hugging Face copy (`databricks/databricks-dolly-15k`) has 15,011 records, so its row numbers differ; the fixed splits and the LLM unit labels of this project belong to the file here.

| category | records | has context |
|---|---|---|
| brainstorming | 1,767 | no |
| classification | 2,136 | no |
| closed_qa | 1,823 | yes |
| creative_writing | 711 | no |
| general_qa | 2,191 | no |
| information_extraction | 1,512 | yes |
| open_qa | 3,611 | no |
| summarization | 1,263 | yes |

**中文。** `databricks-dolly-15k.jsonl` 是 Databricks 发布的 databricks-dolly-15k 原始版本，取自 databrickslabs/dolly 仓库的上述提交，共 15,014 条，原样再分发，许可证为 CC BY-SA 3.0（署名-相同方式共享）。Hugging Face 上的版本是 15,011 条，行号不同；本项目的固定划分和 LLM 单元标签对应这里的文件，`data/dolly.py` 会校验 SHA-256。
