# Scientific Reasoning

不带工具的 Qwen3-4B 评测见 [no_tools_eval/README.md](no_tools_eval/README.md)。支持已下载的 GPQA-Diamond、R-Bench 文本集、SciEval 和 SciBench 文本集，每个数据集最多前 500 题，通过 `nohup` 启动，保留逐题流式思考、答案、评分和断点。

支持通过 vLLM 调用本地部署的 Qwen3，或通过 DeepSeek API 调用远程模型，在科学推理题上循环调用 `search` 和 `code`，直到最终回答包含完整的 `\boxed{...}`。两种方式共用工具定义、调用参数解析、推理循环和结果格式。客户端只依赖 HTTP，不需要安装 vLLM 或模型权重；vLLM 服务可以运行在另一台机器。

## 安装

```bash
cd /fs/archive/share/u2026104455/scientific_reasoning
python -m pip install -e '.[science]'
cp .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`。不需要科学计算库时使用 `pip install -e .`。也可直接使用系统环境变量，环境变量优先于 `.env`。父目录的 `.env` 不会自动加载，可显式指定 `--env-file ../.env`。

## 启动模型

在 GPU 环境安装适合其 CUDA 环境的 vLLM。

本机已有 `.venv` 时先执行 `source .venv/bin/activate`；完整依赖版本保存在 `requirements.txt`，可用 `python -m pip install -r requirements.txt` 重建。

Triton 在启动时需要编译 CUDA 辅助模块，因此还需要与 Python 版本匹配的开发头文件。若出现 `fatal error: Python.h: No such file or directory`，Ubuntu/Debian 可在项目目录执行以下命令，无需 sudo：

```bash
bash scripts/setup_python_headers.sh
```

该脚本从系统配置的软件源下载开发包，仅解压到 `.venv/python-dev`；启动脚本会自动将这些头文件加入编译器的搜索路径。它属于系统开发依赖，无法通过 `requirements.txt` 安装。也可由管理员安装对应版本的 `python3-dev`。

启动模型：

```bash
bash scripts/serve_qwen3.sh /path/to/Qwen3-8B
```

不传模型路径时默认 `/data/shared/Qwen3-8B`，使用 GPU 7，张量并行数为 1。默认服务名为 `Qwen/Qwen3-8B`，替换模型时可用 `SERVED_MODEL_NAME` 设置服务名，并同步修改配置中的 `model.name`。远程服务地址在 `configs/default.yaml` 的 `model.base_url` 中配置。

脚本使用 Qwen3 的 Hermes 工具格式和 `qwen3` 推理解析器，建议使用支持两者的 vLLM 版本（Qwen 官方文档说明 `qwen3` 推理解析器自 0.9.0 起可用）。其他 Qwen3 变体请按各自模型模板选择解析器，可通过脚本末尾参数覆盖。参考 [Qwen vLLM 部署](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)和[工具调用说明](https://qwen.readthedocs.io/en/latest/framework/function_call.html)。

## 使用 DeepSeek API

在 `.env` 或环境变量中填写有效的 `DEEPSEEK_API_KEY`，然后选择 `configs/deepseek.yaml`。此方式不需要启动本地模型服务：

```bash
python -m src --config configs/deepseek.yaml \
  --question 'A 2 kg object moves at 12 m/s. Verify its kinetic energy with code.' \
  --task-name deepseek-kinetic-energy
```

批量推理使用相同的 `--input` 参数；GPQA 全量评测也可直接切换：

```bash
bash scripts/eval_gpqa.sh --config configs/deepseek.yaml --task-name deepseek-gpqa
```

`model.provider` 可选 `vllm` 或 `deepseek`；省略时仍为 `vllm`，原有配置和本地模型启动方式继续可用。仅设置 `provider: deepseek` 时，默认地址为 `https://api.deepseek.com/v1`、模型为 `deepseek-flash`、密钥环境变量为 `DEEPSEEK_API_KEY`；均可通过 `base_url`、`name`、`api_key_env` 覆盖，例如将 `name` 改为 `deepseek-v4-pro`。`configs/deepseek.yaml` 默认开启思考模式，`reasoning_effort: high`；`enable_thinking: false` 可关闭思考，此时不发送 `reasoning_effort`。

DeepSeek 客户端请求 `/chat/completions`，使用与 Qwen3 相同的 `tools` JSON Schema、`tool_choice: auto`、`function.arguments` JSON 字符串和 `role: tool` / `tool_call_id` 回传格式。所有轮次的 `reasoning_content` 原样保留并传入后续请求，包括没有工具调用而需要继续作答的轮次；最终仍只解析回答正文中的 `\boxed{...}`。这是 [DeepSeek 思考模式与工具调用](https://api-docs.deepseek.com/guides/thinking_mode/) 要求的历史传递方式。DeepSeek 请求使用官方 `thinking` 参数，不发送 vLLM 专用的 `top_k` 和 `chat_template_kwargs`；思考模式下不发送无效的 `temperature`。`max_tokens` 限制单轮总生成量，包含思考和回答，截断时仍标记为 `max_tokens`。

DeepSeek 的配置和省略 `max_tokens` 时的默认值均设为 `393216`，即官方当前支持的最大输出 384K tokens（核对日期：2026-10-25，见 [API 参数说明](https://api-docs.deepseek.com/api/create-chat-completion/)）。这是每轮输出上限，不会要求模型生成固定长度；实际长度取决于模型何时结束，输入与输出仍共同受上下文长度限制。本地 vLLM 的生成上限继续使用原有配置。

模型推理由 DeepSeek API 完成，`code` 工具仍在运行本项目的客户端环境中执行，`search` 继续使用原有搜索接口；两者的参数、返回内容和错误回传处理均与本地 Qwen3 方式一致。模型与搜索默认使用同一密钥，也可分别设置 `model.api_key_env` 和 `search.api_key_env`。

## 推理

单题：

```bash
python -m src --config configs/default.yaml \
  --question 'A 2 kg object moves at 12 m/s. Verify its kinetic energy with code.' \
  --task-name kinetic-energy
```

数据集：

```bash
python -m src --config configs/default.yaml \
  --input examples/questions.jsonl --task-name science-questions
```

支持 JSONL 或 JSON 对象数组，默认每题格式：

```json
{"id": "sample-1", "question": "题目内容", "answer": "可选参考答案"}
```

通过 `dataset.question_field` 和 `dataset.id_field` 适配字段名；多选题可将选项一并写入题目字段。参考答案仅随原始记录保存在轨迹中，不发送给模型。`dataset.concurrency` 控制并发题数，同一题内的工具调用按顺序执行。结果按完成顺序逐题写入并刷新，`id` 和 `index` 用于还原输入顺序。每次运行自动新建任务目录。

## 工具与循环

1. 将系统提示词、原始题目及两个工具的 JSON Schema 发给所选模型服务（vLLM 或 DeepSeek API）。
2. 模型返回标准 `tool_calls`，框架执行对应工具。
3. 将 assistant 工具调用和对应的 `role=tool` 结果追加到同一个 `messages` 列表，完整传入下一轮。
4. 没有工具调用且回答正文出现完整、非空的 `\boxed{...}` 时结束。支持嵌套 LaTeX 括号，忽略 `<think>` 内的 box；不检查代码字符串或搜索结果里的 box。
5. 没有 box 时提示模型继续。最大轮数和单轮生成上限用于限制开销，不会把未完成样本标记为成功。

`search` 直接请求 [DeepSeek Search API](https://api-docs.deepseek.com/guides/search/) 的 `POST https://api.deepseek.com/v1/tools/search`，使用 Bearer API Key。模型只传入 `query`（1–128 字符）；项目向 API 发送 `query` 和配置中的 `need_content`（默认 `false`），不发送 `count`。API 返回多少条结果，就按原顺序将多少条完整放入同一个工具结果中，不在客户端限制条数。每条结果只保留 `title`、`url` 和 `text`，其中 `text` 仅取 API 的完整 `summary`，不限制长度、不截断；缺少摘要时为空字符串，结果中不返回 `truncated` 字段。旧配置中的 `max_result_chars` 仅为兼容而接受，不再生效。超时、HTTP 错误和参数错误作为工具结果返回模型，允许修正后再次调用。

`code` 使用当前客户端 Python 环境运行代码；最后一个顶层语句若是表达式，其非 `None` 值会像 Python 交互环境一样自动显示到 `stdout`，因此末尾只写 `total_uncertainty` 也能返回数值。表达式只执行一次；已有的 `print()` 输出保留，末尾的 `print()` 不会额外输出 `None`，赋值语句不会自动显示。已安装的 numpy/scipy/sympy 可直接导入。每次调用启动新解释器，变量不跨调用保留，同一道题内文件可复用，题目结束后临时目录删除。结果包含完整的 `stdout`、`stderr`，以及 `returncode`、`timed_out`；不限制输出长度、不截断，也不返回 `truncated` 字段。旧配置中的 `max_output_chars` 仅为兼容而接受，不再生效。可配置执行超时，超时会终止进程组。子进程不继承父进程的 API Key 环境变量。它是本机进程执行器，并非安全沙箱；运行不可信模型代码时，应将客户端放入权限受限的容器或专用环境。

执行前默认检查代码中的显式 `import`、`from ... import ...`，以及使用字符串字面量的 `importlib.import_module()` / `__import__()`。缺少第三方模块时，先用当前 Python 环境的 pip 安装，再执行用户代码；标准库和已有模块跳过安装。内置常见模块与包名映射，例如 `PIL` → `Pillow`、`sklearn` → `scikit-learn`、`rdkit` → `rdkit`；其他模块默认使用同名 pip 包，也可通过 `code.package_map` 配置映射。只检查顶层模块，已有库中的错误函数名或不存在的子模块仍作为代码错误返回。变量动态计算出的导入名无法预先识别。

`code.auto_install` 默认 `true`；安装超时由 `code.install_timeout` 控制，默认 300 秒，与代码执行的 30 秒分别计算。共享安装锁防止并发题目及同环境下的多个评测进程重复安装。安装成功后，将新增/变更包及其版本更新到 `requirements.txt`，简短安装信息写入工具返回的 `stderr`；安装失败或超时时返回完整安装错误，不执行用户代码。临时安装报告自动清理，不在任务输出目录中增加文件。将 `code.auto_install` 设置为 `false` 可恢复只执行代码的行为。

整个历史消息不会自动裁剪；长题目、多轮搜索可能超出服务的上下文长度，需要按实验规模调整上下文长度、`max_tokens` 或最大轮数。原始推理字段保存在轨迹中，历轮思考如何渲染给模型取决于模型自身的 chat template。

## 输出

所有命令统一保存到项目的 `outputs/<任务名>-<北京时间戳>/`，时间戳格式为 `YYYYMMDD-HHMMSS`，精确到秒。`--task-name` 指定任务名；未指定时，单题为 `single`、数据集为输入文件名、GPQA 全量为 `gpqa-diamond-all`，指定 `--limit N` 时为 `gpqa-diamond-first<N>`。也可用 `--output-dir outputs/<任务名>` 指定目录前缀，自动补上时间戳；不允许保存到项目 `outputs/` 之外。

每个任务目录只保留以下内容：

```text
outputs/<任务名>-<时间戳>/
  traces/         # 每题完整轨迹，随模型响应和工具结果逐步更新
  results.jsonl   # 逐题结果及轨迹路径；GPQA 还包含预测、参考标签和对错
  summary.json    # 运行统计；GPQA 还包含正确率和续跑所需的评测设置
```

每题轨迹保存 `status`、`answer`（box 内的文本）、`rounds`、完整 `messages` 和每轮原始模型消息、`finish_reason`、token usage（`trace`）。通用推理额外保存 `id`、`index` 与原始 `input`。完整对话只存于对应轨迹，不重复写入结果文件。不额外生成题目/答案副本、模型列表、配置或源码副本、CSV 或 Markdown 报告。

状态为 `completed`、`max_rounds`、`max_tokens`；单题异常为 `error`，保留已完成轨迹，其他题继续执行。`max_tokens` 表示本轮输出被截断，框架停止此题，避免执行不完整工具调用；调高单轮生成上限后可重跑。通用推理命令不对参考答案自动评分；GPQA 专用命令支持选项评分，见下文。

## GPQA Diamond 评测

使用 [GPQA 作者官方仓库](https://github.com/idavidrein/gpqa) 发布的 `gpqa_diamond.csv`。本地数据保存在 `datasets/gpqa/`，该目录已加入忽略列表。

```bash
bash scripts/eval_gpqa.sh
```

脚本自动使用项目 `.venv`、`.env` 和 `configs/default.yaml`，默认评测 GPQA Diamond 全部 198 题、选项种子为 42，任务名为 `gpqa-diamond-all`。可从任意目录通过脚本绝对路径启动，无需手动激活环境。追加参数可覆盖默认设置，例如 `bash scripts/eval_gpqa.sh --limit 20 --task-name gpqa-test`。

未设置 `--limit` 时，自动按 CSV 实际行数评测全部题目；设置 `--limit N` 时，只取 CSV 原始顺序的前 N 题，N 必须为正整数且不超过总题数。直接运行 Python 评测入口也遵循此规则，无需在脚本中写死总题数。使用标准 `Question`、`Correct Answer` 和三个 `Incorrect Answer` 字段；仅用固定种子打乱每题选项，不打乱题目顺序。模型只接收题目、选项和单字母 boxed 答案要求，不接收参考答案标签或解析。每题运行一次，按参考选项标签评分，失败、截断或无法解析的答案计错，分母固定为所选题目总数。该命令沿用当前配置，允许搜索和代码工具，因此结果属于使用工具的开卷评测；`--seed` 只控制选项顺序，模型采样未单独设置种子。

`results.jsonl` 运行中按完成顺序追加，结束时按原题序排列；`summary.json` 随每题完成更新正确率，其中 `run` 字段保存数据校验值与评测设置。续跑时使用相同参数，并加上 `--resume --output-dir outputs/<已有任务名-时间戳>`；程序检查数据和评测设置一致后，跳过已有最终结果、继续未完成题目。已失败的最终结果也不会被自动重试。

## SuperGPQA 轨迹合成

使用 DeepSeek API 对本地 `datasets/supergpqa/SuperGPQA-all.jsonl` 合成完整推理轨迹：

```bash
bash scripts/synthesize_supergpqa.sh --start-index 0
```

题号按原始 JSONL 的非空记录从 **0** 开始，最小为 0；当前共 26,529 题，最后一题为 26,528。`--start-index N` 包含第 N 题，默认一直处理到末尾，不打乱题目或选项。`--limit N` 限制处理题数，或用 `--end-index N` 指定不包含该题的结束边界；两者不能同时使用。例：`--start-index 100 --limit 20` 处理题号 100–119。`--concurrency N` 可覆盖默认并发数 4。`--dry-run` 仅校验范围并打印设置，不调用 API。

脚本自动使用项目 `.venv`、`.env` 和 `configs/deepseek.yaml`，沿用现有 search/code 工具和推理循环。提示词包含原始题目及全部选项，参考答案只用于保存和评分，不传入模型。支持 A–J 等全部实际选项标签。

输出保存到 `outputs/supergpqa-deepseek-from<N>-<北京时间戳>/`：`traces/sample-000000.json` 对应原始第 0 题，以此类推；每轮模型响应和工具结果都更新轨迹，保存完整 `messages`、`reasoning_content`、`tool_calls`、工具返回及 token 用量。`results.jsonl` 按完成顺序逐题追加紧凑结果和轨迹路径，`index` 保留原始题号；`summary.json` 持续更新完成数、活动题号、状态、正确率和 token 用量。全部历史轨迹保存在磁盘，内存中只保留每题紧凑结果。

后台启动示例：

```bash
mkdir -p outputs
nohup bash scripts/synthesize_supergpqa.sh --start-index 0 \
  > outputs/supergpqa-deepseek.nohup.log 2>&1 < /dev/null &
echo $! > outputs/supergpqa-deepseek.pid
```

从已有任务续跑时，保留相同的范围及配置，加上 `--resume --output-dir outputs/<已有任务目录>`。已保存的最终结果（包括失败结果）跳过；中断时仍为 `running` 的题目重新执行。

## 目录

```text
configs/                 # YAML 参数
examples/                # 示例数据
scripts/                 # vLLM 服务与数据集评测启动脚本
src/
  clients/               # vLLM、DeepSeek HTTP 客户端与选择入口
  tools/                 # search、code、工具定义与分发
  reasoning/             # 提示词、推理循环、boxed 提取
  data/                  # 数据读写
  config.py              # 配置
  runner.py              # 单题及并发数据集执行
  cli.py                 # 命令行入口
tests/                   # 无需模型/API Key 的测试
```

运行验证：

```bash
python -m unittest discover -s tests -v
```

测试用模拟 HTTP 验证 vLLM 和 DeepSeek 的搜索→真实 Python 计算→最终 boxed 完整循环，并覆盖请求参数差异、思考历史回传、多工具、错误回传、轮数/输出上限、代码超时、字段映射与并发。真实模型质量、GPU 部署和 DeepSeek 账号访问需在配置服务与 API Key 后验证。
