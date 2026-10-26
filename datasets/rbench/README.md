---
dataset_info:
- config_name: rbench-m_en
  features:
  - name: index
    dtype: string
  - name: image
    dtype: image
  - name: question
    dtype: string
  - name: A
    dtype: string
  - name: B
    dtype: string
  - name: C
    dtype: string
  - name: D
    dtype: string
  - name: E
    dtype: string
  - name: F
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test
    num_bytes: 53437023.0
    num_examples: 665
  download_size: 53105849
  dataset_size: 53437023.0
- config_name: rbench-m_zh
  features:
  - name: index
    dtype: string
  - name: image
    dtype: image
  - name: question
    dtype: string
  - name: A
    dtype: string
  - name: B
    dtype: string
  - name: C
    dtype: string
  - name: D
    dtype: string
  - name: E
    dtype: string
  - name: F
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test
    num_bytes: 51548006.0
    num_examples: 665
  download_size: 51317105
  dataset_size: 51548006.0
- config_name: rbench-t_en
  features:
  - name: index
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: A
    dtype: string
  - name: B
    dtype: string
  - name: C
    dtype: string
  - name: D
    dtype: string
  - name: E
    dtype: string
  - name: F
    dtype: string
  splits:
  - name: test
    num_bytes: 686920
    num_examples: 1094
  download_size: 369218
  dataset_size: 686920
- config_name: rbench-t_zh
  features:
  - name: index
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: A
    dtype: string
  - name: B
    dtype: string
  - name: C
    dtype: string
  - name: D
    dtype: string
  - name: E
    dtype: string
  - name: F
    dtype: string
  splits:
  - name: test
    num_bytes: 607983
    num_examples: 1094
  download_size: 358002
  dataset_size: 607983
configs:
- config_name: rbench-m_en
  data_files:
  - split: test
    path: rbench-m_en/test-*
- config_name: rbench-m_zh
  data_files:
  - split: test
    path: rbench-m_zh/test-*
- config_name: rbench-t_en
  data_files:
  - split: test
    path: rbench-t_en/test-*
- config_name: rbench-t_zh
  data_files:
  - split: test
    path: rbench-t_zh/test-*
license: apache-2.0
task_categories:
- question-answering
- text2text-generation
language:
- en
- zh
tags:
- chemistry
- biology
- legal
- finance
- code
- climate
- medical
size_categories:
- 1K<n<10K
---

# R-Bench

## Introduction


[R-Bench](https://evalmodels.github.io/rbench/) is a graduate-level multi-disciplinary benchmark for evaluating the complex reasoning capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). R stands for Reasoning.

According to statistics on R-Bench, the benchmark spans 19 departments, including mathematics, physics, biology, computer science, and chemistry, covering over 100 subjects such as Inorganic Chemistry, Chemical Reaction Kinetics, and Electromagnetism. It features 1,094 questions designed for testing language models and 665 questions specifically tailored for evaluating multimodal reasoning capabilities, available in both English and Chinese.

These questions are meticulously curated to ensure rigorous difficulty calibration, subject balance, and cross-linguistic alignment, enabling the assessment to be an Olympiad-level multi-disciplinary benchmark.

## Official Links

### Paper

[R-Bench: Graduate-level Multi-disciplinary Benchmarks for LLM & MLLM Complex Reasoning Evaluation](https://arxiv.org/abs/2505.02018)

## Evaluation Results

### Language Model Results


| Model                      | Source                                                                 | Date       | Average | RBench-T | RBench-T (zh) |
|---------------------------|------------------------------------------------------------------------|------------|---------|----------|----------------|
| OpenAI o1 🥇               | [link](https://openai.com/o1/)                                         | 2024-12-17 | 69.6    | 69.0     | 70.1           |
| Gemini2.0-Flash-Thinking 🥈 | [link](https://deepmind.google/technologies/gemini/flash-thinking/)    | 2025-01-21 | 68.0    | 68.4     | 67.5           |
| Doubao1.5Pro 🥉            | [link](https://www.volcengine.com/product/doubao)                      | 2025-01-21 | 62.7    | 62.0     | 63.4           |
| GPT-4o                     | [link](https://openai.com/index/hello-gpt-4o/)                          | 2024-11-20 | 52.6    | 53.6     | 51.6           |
| Claude3.5-sonnet           | [link](https://www.anthropic.com/news/claude-3-5-sonnet)               | 2024-06-20 | 57.4    | 57.5     | 57.3           |
| Qwen2.5-72B                | [link](https://github.com/QwenLM/Qwen2.5)                              | 2024-09-19 | 52.9    | 53.7     | 52.0           |
| Qwen2.5-32B                | [link](https://github.com/QwenLM/Qwen2.5)                              | 2024-09-19 | 50.4    | 50.8     | 49.9           |
| Qwen2.5-7B                 | [link](https://github.com/QwenLM/Qwen2.5)                              | 2024-09-19 | 44.1    | 43.6     | 44.5           |


### Multimodal Model Results


| Model                  | Source                                               | Date       | Average | RBench-M | RBench-M (zh) |
|------------------------|------------------------------------------------------|------------|---------|----------|----------------|
| OpenAI o1 🥇            | [link](https://openai.com/o1/)                      | 2024-12-17 | 53.1    | 53.2     | 53.0           |
| Doubao1.5Pro 🥈         | [link](https://www.volcengine.com/product/doubao)   | 2025-01-21 | 40.2    | 37.9     | 42.4           |
| Claude-3-5-sonnet 🥉    | [link](https://www.anthropic.com/news/claude-3-5-sonnet) | 2025-04-10 | 39.0    | 39.7     | 38.3           |
| GPT-4o                 | [link](https://openai.com/index/hello-gpt-4o/)       | 2024-11-20 | 33.3    | 33.4     | 33.2           |
| Qwen2.5-72B            | [link](https://github.com/QwenLM/Qwen2.5)           | 2024-09-19 | 25.4    | 25.1     | 25.7           |
| Qwen2.5-7B             | [link](https://github.com/QwenLM/Qwen2.5)           | 2024-09-19 | 21.0    | 19.6     | 22.3           |


Note:
- RBench-T: Text-only questions for language models test
- RBench-M: Multimodal questions for multimodal models test
- The values in the table represent the Top-1 accuracy, in %
- zh indicates the Chinese version

## Reference

```
@inproceedings{
  guo2025rbench,
  title={RBench: Graduate-level Multi-disciplinary Benchmarks for
    LLM & MLLM Complex Reasoning Evaluation},
  author={Meng-Hao Guo, Jiajun Xu, Yi Zhang, Jiaxi Song, Haoyang Peng, Yi-Xuan Deng, 
    Xinzhi Dong, Kiyohiro Nakayama, Zhengyang Geng, Chen Wang, Bolin Ni, Guo-Wei Yang, 
    Yongming Rao, Houwen Peng, Han Hu, Gordon Wetzstein, Shi-min Hu},
  year={2025},
  eprint={2505.02018},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.02018}, 
}