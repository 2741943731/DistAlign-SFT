# NuminaMath-CoT 单卡 SFT 试跑

这组实验为 IDFT 表 8 的 SFT 对照做准备。原文未公布 1.5B 的完整训练配置；公开参考是 [DFT §4.1.1](https://arxiv.org/html/2508.05629v2#S4.SS1.SSS1) 用 NuminaMath-CoT、2048 token、有效 batch 256、AdamW `5e-5`、余弦衰减及 10% 预热。本文档的 10,240 条候选和单卡 LoRA 是独立试跑的预算，不能与论文的 100,000 条训练样本等同。

在解包目录中，`bash scripts/run_numina_sft_pilot.sh probe` 用原有训练文件中最长的 8 条合格样本做 2 次 SFT 参数更新。训练使用服务器已有的 `nanovllm` Python、原始 Qwen2.5-Math-1.5B 权重和第七张 L40（GPU 索引 6）。观察实际最长序列和 `run.json` 中的 PyTorch 显存峰值；旧样本最长不足 2048 时，这个峰值不能代表完整长度的最坏情况。试跑产生的 adapter 不用来报告分数。

`bash scripts/run_numina_sft_pilot.sh setup-data` 创建独立的数据环境，保留已有训练和推理环境。`bash scripts/run_numina_sft_pilot.sh prepare` 在服务器从 [NuminaMath-CoT](https://huggingface.co/datasets/project-numina/NuminaMath-CoT) 的固定版本 `9d8d210c9f6a36c8f3cd84045668c9b7800ef517` 流式读取首片最多 16,384 行，只取题目、解答和来源。按来源标记、规范化后的题面和重复项过滤，再按种子 42 的稳定顺序切出 10,240 条训练候选和 256 条开发候选。这个前缀不代表整库随机样本。随包提供的 MATH-500 文件只有 500 道题面，用于精确去重；近重复、改写和翻译仍可能漏过。

数据成功时查看 `results/data/manifest.json` 和 `results/data/split/audit.json`。开发集的教师答案需独立核验后才能用来选参数；训练候选还需用固定 tokenizer 检查 2048 token 上限，实际训练行数可能更少。如筛后不足，保留此次输出，在新解包目录设置 `DISTALIGN_CANDIDATE_LIMIT` 重试，并记录新的上限。

下一阶段拟用一轮 SFT、microbatch 1、梯度累积 256、上述优化设置和相同 LoRA 结构；最终设置根据显存试跑和实际入选样本数确认。DFT 和 IDFT 后续应使用同一训练切分、同一开发集、同一模型权重和同等学习率搜索预算。冻结设置后才在 MATH-500 上跑 v2 正式评测，比较项目已测 Base **48.8625%**。
