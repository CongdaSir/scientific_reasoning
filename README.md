# Scientific Reasoning

通过 vLLM 部署 Qwen3，在科学推理题上循环调用 `search` 和 `code`，直到最终回答包含完整的 `\boxed{...}`。客户端只依赖 HTTP，不需要安装 vLLM 或模型权重；vLLM 服务可以运行在另一台机器。

## 安装

```bash
cd /fs/archive/share/u2026104455/scientific_reasoning
python -m pip install -e '.[science]'
cp .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`。不需要科学计算库时使用 `pip install -e .`。也可直接使用系统环境变量，环境变量优先于 `.env`。父目录的 `.env` 不会自动加载，可显式指定 `--env-file ../.env`。

## 启动模型

在 GPU 环境单独安装适合其 CUDA 环境的 vLLM，再启动：

```bash
bash scripts/serve_qwen3.sh /path/to/Qwen3-8B --tensor-parallel-size 2
```

不传模型路径时默认 `Qwen/Qwen3-8B`。默认服务名也是 `Qwen/Qwen3-8B`，替换模型时可用 `SERVED_MODEL_NAME` 设置服务名，并同步修改配置中的 `model.name`。远程服务地址在 `configs/default.yaml` 的 `model.base_url` 中配置。

脚本使用 Qwen3 的 Hermes 工具格式和 `qwen3` 推理解析器，建议使用支持两者的 vLLM 版本（Qwen 官方文档说明 `qwen3` 推理解析器自 0.9.0 起可用）。其他 Qwen3 变体请按各自模型模板选择解析器，可通过脚本末尾参数覆盖。参考 [Qwen vLLM 部署](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)和[工具调用说明](https://qwen.readthedocs.io/en/latest/framework/function_call.html)。

## 推理

单题：

```bash
python -m src --config configs/default.yaml \
  --question 'A 2 kg object moves at 12 m/s. Verify its kinetic energy with code.' \
  --output outputs/single.jsonl
```

数据集：

```bash
python -m src --config configs/default.yaml \
  --input examples/questions.jsonl --output outputs/results.jsonl
```

支持 JSONL 或 JSON 对象数组，默认每题格式：

```json
{"id": "sample-1", "question": "题目内容", "answer": "可选参考答案"}
```

通过 `dataset.question_field` 和 `dataset.id_field` 适配字段名；多选题可将选项一并写入题目字段。参考答案仅随原始记录保存，不发送给模型。`dataset.concurrency` 控制并发题数，同一题内的工具调用按顺序执行。文件按完成顺序逐题写入并刷新，`id` 和 `index` 用于还原输入顺序。每次运行会覆盖指定输出文件。

## 工具与循环

1. 将系统提示词、原始题目及两个工具的 JSON Schema 发给 vLLM。
2. 模型返回标准 `tool_calls`，框架执行对应工具。
3. 将 assistant 工具调用和对应的 `role=tool` 结果追加到同一个 `messages` 列表，完整传入下一轮。
4. 没有工具调用且回答正文出现完整、非空的 `\boxed{...}` 时结束。支持嵌套 LaTeX 括号，忽略 `<think>` 内的 box；不检查代码字符串或搜索结果里的 box。
5. 没有 box 时提示模型继续。最大轮数和单轮生成上限用于限制开销，不会把未完成样本标记为成功。

`search` 直接请求 [DeepSeek Search API](https://api-docs.deepseek.com/guides/search/) 的 `POST https://api.deepseek.com/v1/tools/search`，使用 Bearer API Key，参数为 `query`、`count` 和 `need_content`。模型可设置 query（1–128 字符）与 count（1–50）；未指定 count 时使用配置值。默认将标题、链接和长摘要返回模型；开启 `need_content` 后优先返回正文。`max_result_chars` 限制每条正文/摘要长度，并在结果中标明截断。超时、HTTP 错误和参数错误作为工具结果返回模型，允许修正后再次调用。

`code` 使用当前客户端 Python 环境运行代码，以 `print()` 返回计算结果；已安装的 numpy/scipy/sympy 可直接导入。每次调用启动新解释器，变量不跨调用保留，同一道题内文件可复用，题目结束后临时目录删除。配置执行超时及输出长度上限，超时会终止进程组。子进程不继承父进程的 API Key 环境变量。它是本机进程执行器，并非安全沙箱；运行不可信模型代码时，应将客户端放入权限受限的容器或专用环境。

整个历史消息不会自动裁剪；长题目、多轮搜索可能超出服务的上下文长度，需要按实验规模调整上下文长度、`max_tokens`、搜索结果长度或最大轮数。原始推理字段保存在轨迹中，历轮思考如何渲染给模型取决于模型自身的 chat template。

## 输出

每题保存 `status`、`answer`（box 内的文本）、`rounds`、完整 `messages` 和每轮原始模型消息、`finish_reason`、token usage（`trace`）。批量模式额外保存 `id`、`index` 与原始 `input`。

状态为 `completed`、`max_rounds`、`max_tokens`；批量模式中的单题异常为 `error`，其他题继续执行。`max_tokens` 表示本轮输出被截断，框架停止此题，避免执行不完整工具调用；调高单轮生成上限后可重跑。异常样本保存错误和原始输入，不保证保存中断前的轨迹。本项目只负责推理与轨迹收集，不对参考答案自动评分。

## 目录

```text
configs/                 # YAML 参数
examples/                # 示例数据
scripts/                 # vLLM 启动脚本
src/
  clients/               # vLLM HTTP 客户端
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

测试用模拟 HTTP 验证搜索→真实 Python 计算→最终 boxed 的完整循环，并覆盖多工具、错误回传、轮数/输出上限、代码超时、字段映射与并发。真实模型质量、GPU 部署和 DeepSeek 账号访问需在配置服务与 API Key 后验证。
