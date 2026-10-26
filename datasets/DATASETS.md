# 科学推理数据集

准备日期：2026-10-25（北京时间）。所有文件保留官方发布格式，未合并或修改题目。

| 数据集 | 本地目录 | 主要文件和范围 | 官方来源 |
| --- | --- | --- | --- |
| GPQA-Diamond | `gpqa/` | `gpqa_diamond.csv`，198 题 | https://github.com/idavidrein/gpqa |
| SuperGPQA | `supergpqa/` | `SuperGPQA-all.jsonl`，26,529 题 | https://huggingface.co/datasets/m-a-p/SuperGPQA |
| R-Bench | `rbench/` | 中英文文本与多模态四个子集，Parquet 格式，包含答案和多模态图像 | https://huggingface.co/datasets/R-Bench/R-Bench |
| SciEval | `scieval/` | 开发集、验证集、测试集、带答案本地测试集及动态化学/物理数据 | https://github.com/OpenDFM/SciEval |
| SciBench | `scibench/` | 官方 `dataset/` 全目录，包含文本题、解答示例、工具提示和图像题 | https://github.com/mandyyyyii/scibench |

## 文件位置与数量

- GPQA-Diamond：`gpqa/gpqa_diamond.csv`。沿用已有文件，并从记录的官方固定版本重新获取压缩包进行逐字节比对；CSV 的 SHA-256、题数及压缩包成员的大小和 CRC32 均通过检查。原有文件未覆盖。
- SuperGPQA：`supergpqa/SuperGPQA-all.jsonl`，26,529 条记录。
- R-Bench：`rbench/rbench-t_en/`、`rbench/rbench-t_zh/`、`rbench/rbench-m_en/`、`rbench/rbench-m_zh/`，各含 `test-00000-of-00001.parquet`。官方 README 标注文本子集每种语言 1,094 题，多模态子集每种语言 665 题；两种语言是对应版本。
- SciEval：`scieval/scieval-dev.json`（70 条）、`scieval/scieval-valid.json`（1,441 条）、`scieval/scieval-test.json`（13,011 条）、`scieval/scieval-test-local.json`（13,011 条，带答案）、`scieval/dynamic_chem.json`（2,000 条）、`scieval/dynamic_phy.json`（890 条）。两个 test 文件是同一测试集的不同版本，不应重复计数。
- SciBench：`scibench/dataset/original/` 的 10 个非 `_sol` JSON 文件共 580 题，10 个 `_sol.json` 文件共 112 条解答示例；`scibench/dataset/img/problem/` 含 151 道图像题及 PNG 文件，`scibench/dataset/img/problem_solution/` 含 26 条图像解答示例。下载的是官方当前主分支数据，旧版本位于上游 `old` 分支。

## 固定版本

| 数据集 | 仓库版本 |
| --- | --- |
| GPQA-Diamond | `56686c06f5e19865c153de0fdb11be3890014df7` |
| SuperGPQA | `4430d4458112c7d4497fdcf94d7cc223313d6acf` |
| R-Bench | `c0e92a6d90dad980ea51a47af64c12ec67b670ce` |
| SciEval | `aa61a9dfe75f55145492517fac0b7f05220b0d72` |
| SciBench | `e14e0ca3d3db0f02928a18d435b9c3d15656a7c5` |

每个数据集目录内的 `download_manifest.json` 记录来源 URL、固定版本、文件大小、SHA-256 和校验方法。Hugging Face 大文件已与上游 LFS SHA-256 比对，其他新下载文件已与上游 Git blob SHA-1 比对。JSON/JSONL 已解析检查，Parquet 已检查文件头尾标记及官方文件校验值。R-Bench 题数引用下载的官方 README，未安装额外依赖解码 Parquet。

各目录保留官方 README；GPQA 的 `license.txt` 与 SciBench 的 `LICENSE` 也已保留。引用信息见对应官方 README。
