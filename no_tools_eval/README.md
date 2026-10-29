# Qwen3-4B 无工具评测

独立的单轮、零样本评测流程。每题只请求一次模型回答，不加载原项目的工具注册表、推理循环、搜索服务或 Python 执行器，HTTP 请求不包含 `tools`、`tool_choice`、`functions`。不需要 API Key 或 `.env`。

保留 Qwen3 的 thinking 模式；“无工具”不等于关闭模型思考。参考答案、题目解析和 SciBench 解答示例不进入模型对话。

## 默认评测范围

| 数据集 | 本地数据 | 可用题数 | 评测题数 |
| --- | --- | ---: | ---: |
| GPQA-Diamond | `datasets/gpqa/gpqa_diamond.csv` | 198 | 198 |
| R-Bench | `rbench-t_en` 英文文本测试集 | 1,094 | 500 |
| SciEval | `scieval-test-local.json` 带答案测试集 | 13,011 | 500 |
| SciBench | `dataset/original/` 的 10 个非 `_sol` JSON | 580 | 500 |
| 合计 | | | **1,698** |

保留源文件行序，不随机抽题。GPQA 的选项使用种子 42 打乱，与原项目 GPQA 评测保持一致。SciBench 按文件名排序后按行拼接，在整体序列上取前 500 题；文件顺序写入运行清单。

R-Bench 默认使用英文版本；`--rbench-subset rbench-t_zh` 可切换为对应中文版本，两个语言版本不合并计数。Qwen3-4B 是文本生成模型，因此 R-Bench 多模态子集及 SciBench 图像题不评测，清单记录排除原因。

下载的 R-Bench 第 236 条（源 `index=236`）在中英文版本中都将参考答案写为两行 `E`、`F`，不符合单选标签格式。该条仍保留在前 500 题中并生成回答，记录原始参考答案和 `is_correct=null`，从准确率分母排除；因此本次 R-Bench 生成 500 条，最多有 499 条可评分。原始数据不修改，问题位置记录在 `data_quality.json`。

SciEval 原始测试集前 500 题全部属于 MedQA，不能把这次截断结果理解为各学科的均衡评测。开发集、验证集、动态集及不带答案的重复测试集不混入。

## 启动和续跑

在项目根目录执行：

```bash
# 校验数据、评分参考和上下文长度，不请求模型、不创建结果文件。
.venv/bin/python -m no_tools_eval --dry-run

# 使用 nohup 启动模型服务和评测。
bash no_tools_eval/launch.sh

# 指定 GPU、端口、并发和输出目录。
GPU=2 PORT=18041 CONCURRENCY=16 bash no_tools_eval/launch.sh outputs/my-no-tools-run

# 同目录续跑，已完成题目不重复请求。
bash no_tools_eval/launch.sh outputs/my-no-tools-run --resume

# 追加 --retry-errors 可重新请求上次最终报错的题目。
bash no_tools_eval/launch.sh outputs/my-no-tools-run --resume --retry-errors
```

脚本使用项目 `.venv`，可从任意目录通过脚本绝对路径启动。默认从 `/data/shared/model/Qwen3-4B` 离线加载权重，使用 GPU 2、端口 18041、并发 16。服务仅监听 `127.0.0.1`。可以设置 `MODEL_PATH`、`MODEL_NAME`、`GPU_MEMORY_UTILIZATION`；默认单 GPU、bfloat16、55% GPU 显存、40,960 上下文、`qwen3` 思考解析器，未开启工具调用解析。

健康服务且模型名相符时复用；端口被其他服务占用时退出。每个结果目录持有独占锁，避免两个评测进程同时写入。`--resume` 校验数据哈希、选题和生成参数，参数不相同时拒绝续跑。中断题目的旧流文件会保留，新请求使用新的 attempt 文件。退出评测不会关闭独立的模型服务，可通过 PID 单独停止。

## 生成和评分

默认 `temperature=0.6, top_p=0.95, top_k=20, min_p=0`，启用 thinking，每题输出上限 32,768 token，采样种子为 `42 + 题目索引`。启动前用本地 tokenizer 检查每题输入加输出上限是否超出上下文。参数依据 [Qwen3-4B 模型说明](https://huggingface.co/Qwen/Qwen3-4B#best-practices)。

- 选择题：提取最终回答最后一个完整的 `\boxed{...}`，也支持明确的 JSON `answer`、最终答案标记或独立选项字母，按标签精确评分。思考字段中的答案不参与评分。
- SciBench：要求模型在指定单位下给出数值，预先将原单位中的 `10^n` 倍率移入参考数值；采用[官方数值评分的 5% 相对容差](https://github.com/mandyyyyii/scibench/blob/e14e0ca3d3db0f02928a18d435b9c3d15656a7c5/eval/eval_zeroCot.py)，绝对容差为 0。支持科学计数法、分数和平方根，以受限算术 AST 解析最终数值，不执行模型代码。
- SciEval 非选择题适配遵循[官方参考答案匹配方式](https://github.com/OpenDFM/SciEval/blob/aa61a9dfe75f55145492517fac0b7f05220b0d72/eval.py)；本次前 500 题均为选择题。
- 截断、异常和无法解析的回答计为错误。只在连接/流失败、HTTP 429 或 5xx 时最多重试两次；不会因答案错误或截断追加采样，不使用模型裁判。

## 过程与结果

每次创建 `outputs/no-tools-qwen3-4b-<北京时间>/`：

```text
manifest.json                  # 数据 SHA-256、协议、生成参数、代码哈希
data_quality.json              # 无效参考答案的位置和排除原因
prepared.jsonl                 # 固定选题、顺序、选项及本地评分参考
server.pid / server.log        # 模型服务 PID 和日志（本次启动服务时）
evaluator.pid / nohup.log      # 评测 PID 和实时日志
summary.json                   # 全部数据集实时汇总
<dataset>/summary.json         # 当前数据集及分学科统计
<dataset>/results.jsonl        # 原始索引顺序的逐题评分和 trace 路径
<dataset>/traces/<id>.json     # 请求、完整思考和答案、token 数、状态、评分
<dataset>/streams/<id>.attempt-<n>.jsonl  # 每次请求的原始流式数据帧
```

收到的每个流式数据帧立即写入文件，生成中的 trace 每约 2 秒原子更新；每题结束后更新结果和汇总。断开连接时已有流和部分思考仍保留。记录使用北京时间，保存开始、结束、耗时和请求次数。

`accuracy_on_finished` 分母是已结束且可评分的题数 `scored`；`accuracy_over_selected` 分母是选定题目中可评分的题数 `scorable_total`。`total` 和 `finished` 仍包含所有选定题目，`data_errors` 记录源参考答案异常数。运行期间结合 `finished/pending/status` 阅读，最终成绩以结束后的各数据集汇总为准。续跑时保存代码哈希历史，已有完整回答继续复用。

```bash
tail -f outputs/<运行目录>/nohup.log
cat outputs/<运行目录>/summary.json

# 停止自己启动的评测或服务。
kill "$(cat outputs/<运行目录>/evaluator.pid)"
kill "$(cat outputs/<运行目录>/server.pid)"
```

验证：`.venv/bin/python -m unittest discover -s tests -v`。
