# SuperGPQA 原题答案检索核对

分析对象：`/storage/users/congdaxiao/scientific_reasoning/outputs/supergpqa-deepseek-from0-20261025-093955`。所有题号从 0 开始。仅检查保存结果，未重启推理或访问网页。

115 道按本地参考标签答对的题目中，61 道使用过搜索。逐条核对这 61 道后，保守确认至少 **20 道**的搜索结果直接提供了原题答案或现成解答，占答对题目的 **17.39%**。其中 8 道来自数据集或基准题解页面，12 道来自普通考试或问答网站。

“通过搜索找到答案”在本报告中指可观察到的来源和后续使用，不代表证明模型在没有搜索时一定无法答对。正确仅指匹配本地参考标签。

| 分类 | 数量 | 说明 |
| --- | ---: | --- |
| 原题答案或现成解答明确可见 | 20 | 后续模型引用该答案/解答，且匹配本地参考标签 |
| 原题为多选，当前数据要求单选 | 4 | 查到多个正确项，无法直接确定本地唯一选项；不计入 20 |
| 找到原题或数据片段，但答案证据不足 | 7 | 未显示明确答案，或只有相邻/相似题解；不计入 20 |
| 其他搜索后答对 | 30 | 主要为知识查询、教材说明或未找到原题答案 |

前一版报告中的 21 条是“结果含数据集链接”的自动标记，并非直接找到答案的数量。本次同时核对普通中文题库、原始考试解析和数据集页面。前一版的 92 条候选只是初筛，不能当作已经排除所有答案检索的干净推理样本。

## 可确认的 20 道

0, 1, 4, 5, 11, 19, 20, 31, 40, 41, 50, 58, 71, 81, 83, 88, 96, 105, 126, 143

| 题号 | 本地答案 | 来源 | 核对说明 |
| ---: | --- | --- | --- |
| 0 | I | [保存的网页来源](https://www.kaggle.com/datasets/avnguyen32/supergpqa-dataset?query=huffman#1) | 同题数据集条目明确列出答案文本及 answer_letter=I。 |
| 1 | G | [保存的网页来源](https://huggingface.co/datasets/alexshengzhili/supergpra-filtered-round2?view=table#1) | 同题题干、十个选项后直接给出正确答案文本，对应本地 G。 |
| 4 | G | [保存的网页来源](https://huggingface.co/ScienceOne-AI/HiSciBench/commit/8da7235ae451f6d14e23a766b47d6138c710429e.diff#15) | 检索到相同棱柱题目及基准题解，题解明确给出 3! = 6；模型据此选择 G。题解摘要不含完整题干，需结合本轨迹前一条同题检索结果核对。 |
| 5 | G | [保存的网页来源](https://huggingface.co/datasets/Siyuc/SER/commit/d25ed942d748f84d07db5cd971f85a5e42d044c4.diff?file=cache-data%2Fpreprocessed%2Fbenchmarks%2Fsupergpqa_science_full.json#1) | 同题数据集 ground_truth 明确给出两次能量沉积同时发生，对应 G。 |
| 11 | F | [保存的网页来源](https://huggingface.co/datasets/sleeping-ai/SupriseBench/commit/c32a91c738c8dd633040ef83e7db3534212b4d87.diff?file=eval_dataset.jsonl#9) | 完整同题选项后明确列出 answer=F；模型明确引用该标签。 |
| 19 | J | [保存的网页来源](http://m.tj.offcn.com/html/2019/09/206256.html) | 同题参考解答给出入账价值 697 万元，对应本地 J。 |
| 20 | D | [保存的网页来源](https://www.lawdun.com/lawn/d72633.html) | 同题借款合同及日期，原题明确标注 D，与本地 D 对应。 |
| 31 | B | [保存的网页来源](http://www.91exam.org/gk/493-1744/1744128-10.html) | 同题俄国革命引文的解答明确选择国家干预，对应本地 B。 |
| 40 | E | [保存的网页来源](https://m.gx.huatu.com/2022/1225/1868839.html) | 基本相同的大学毕业生权利题，原题答案为选举权与被选举权，对应 E；原网页年龄为 24 岁，本地为 23 岁。 |
| 41 | G | [保存的网页来源](https://huggingface.co/datasets/m-a-p/SuperGPQA-Records/viewer/default/train?sql=SELECT+*+FROM+train+WHERE+question+LIKE+%27%25polymer%25%27+LIMIT+10#4) | 同题数据集推理记录明确给出 G 及商品价值的正确答案文本。 |
| 50 | D | [保存的网页来源](https://www.shititong.cn/cha-kan/shiti/0005fee3-0b5c-6c58-c03a-3415160a2c0b.html) | 同题二元论根本错误，网页明确给出否认世界统一性，对应本地 D。 |
| 58 | F | [保存的网页来源](https://www.fentiku.com/wenzhi/4474511/) | 同题国家垄断资本主义，明确答案为政府调控宏观经济，对应 F。 |
| 71 | B | [保存的网页来源](https://huggingface.co/datasets/Siyuc/SER/commit/d25ed942d748f84d07db5cd971f85a5e42d044c4.diff?file=cache-data%2Fpreprocessed%2Fbenchmarks%2Fsupergpqa_science_full.json#4) | 同题谐振子波函数公式的 ground_truth，与本地 B 一致。仅指本地标签一致，未判定公式正确性。 |
| 81 | D | [保存的网页来源](https://www.zuotishi.com/yikao/7583/t2494657.html) | 同题气厚性能，原题明确 B 助阳发热，对应本地 D。 |
| 83 | C | [保存的网页来源](https://www.etest8.com/sifa/zhinan/4604041.html) | 同一公民与医生案例的完整解答明确选 D，对应本地 C。仅核对答案来源，不评价所引历史法律解释的现行效力。 |
| 88 | J | [保存的网页来源](http://ganzhou.htkaoyan.com/show/14389.html) | 同题按劳分配实现形式，现成解析明确为货币形式，对应 J。 |
| 96 | D | [保存的网页来源](https://huggingface.co/datasets/Siyuc/SER/commit/d25ed942d748f84d07db5cd971f85a5e42d044c4.diff?file=cache-data%2Fpreprocessed%2Fbenchmarks%2Fsupergpqa_science_full.json#5) | 第二次检索片段明确列出同题 ground_truth，对应 D；首次返回只有选项，不足以单独确认。 |
| 105 | B | [保存的网页来源](https://m.360kao.com/shiti/5055.html) | 同题化脓性脑膜炎合并硬膜下积液，题解明确病原菌答案，对应本地 B。 |
| 126 | E | [保存的网页来源](https://www.tihai5.com/timu/63450860xcpu.html) | 同题宋明理学根本出发点，明确答案为规范秩序维护统治，对应本地 E。 |
| 143 | E | [保存的网页来源](https://page.sm.cn/blm/midpage-317/index?id=13_8374ff5e2248b11acfabbeb7cc6c0be8&q=sinx-tanx%e6%98%af%e5%81%b6%e5%87%bd%e6%95%b0%e5%90%97&title=1.%e8%ae%bef%ef%bc%88x%ef%bc%89x%2atanx%2ae%5esinx%2c%e5%88%99f%ef%bc%88x%ef%bc%89%e6%98%af%28%29A.%e5%81%b6%e5%87%bd%e6%95%b0+B.%e6%97%a0%e7%95%8c%e5%87%bd%e6%95%b0+C.%e5%91%a8%e6%9c%9f%e5%87%bd%e6%95%b0+D.%e5%8d%95%e8%b0%83%e5%87%bd%e6%95%b0) | 完全相同函数与原始四选项，网页解答明确选无界函数，对应本地 E。 |

## 原题多选的 4 道

- 第 69 题：原始政治题为多选，检索答案含多个正确命题；本地参考 I 只是其中之一，不能据网页确认唯一单选答案。
- 第 104 题：原题支票案例为多选，网上答案 BD；本地 G、E 分别对应原题两个正确项，不能确认唯一单选答案。
- 第 116 题：原题共产主义实证分析为多选，网上答案 ABD；本地单选参考只取其中一个命题。
- 第 149 题：原题债券投资风险为多选，利率、汇率、流动性均在答案内；本地 B 只取利率风险，模型明确讨论了歧义。

## 严格统计暂不计入的 7 道

- 第 26 题：找到了完全相同病例及检查选项，但保存的同题片段未显示目标小题的正确答案；另有相关病例答案不能替代。
- 第 57 题：同题数据集 diff 片段结尾出现 E，但答案字段被省略且前面有相邻题 answer=F；模型据片段推定 E，严格统计不计入。
- 第 90 题：找到高度相似的症状题和另一病例的诊断解答，但同题片段只显示选项，不能严格确认原题答案已显示。
- 第 92 题：找到完全相同相机案例和原四选项，模型称原题答案 B，但保存网页片段没有明确答案或解答。
- 第 117 题：找到相同诊断题及原选项，模型声称原题答案 E，但保存搜索结果没有明确答案字段。
- 第 127 题：找到完全相同 Gauss 积分题及四选项，保存同题页面没有明确答案；其他教材说明非线性方程组，不计为直接读取原题答案。
- 第 144 题：找到完全相同 LH 高峰题及选项，页面在“正确答案:”处结束，答案缺失；模型称 likely B，不能确认直接读取答案。

完整 61 道逐题分类、20 道的来源 URL、搜索返回位置、证据片段、最终思考以及原始轨迹哈希保存在 `answer_lookup_audit.csv`。汇总见 `answer_lookup_summary.json`。搜索返回位置使用 messages 和 results 数组的 0 起始下标。

少数已计入题目包含无关条件的小幅改写（例如第 40 题年龄 23/24 岁）；第 4 题需要结合连续两次检索的同题命中与题解摘要核对，具体限制已逐条列明。
