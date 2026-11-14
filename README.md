# DistAlign-SFT

研究小模型的数学推理后训练：在同一份教师 CoT 上比较 SFT、DFT、IDFT，再通过 Hinted Decoding 调整数据分布，检查数据与损失的交互如何影响泛化。[原始项目方案与参考文献](docs/DistAlign-SFT_项目方案与参考文献.md)保留了完整实验设计。

当前是 Phase 0 起点：已包含本地 JSONL 清洗、SFT / DFT 损失，以及一条用小型随机初始化模型运行的 CPU 冒烟流程。尚未使用 Qwen 或真实训练数据跑实验。[实验登记表](experiments/ablations.md)记录计划中的 A–E 组。

## 本地验证

使用 Python 3.11 或更新版本及 PyTorch。依赖安装后，示例不需要下载数据或模型。以下命令在项目根目录运行：

```bash
uv sync --python 3.12
uv run python -m data.prepare --input examples/teacher.jsonl \
  --held-out examples/heldout.jsonl --output-dir data/processed/demo
uv run python -m src.train --config configs/sft.toml --train-data data/processed/demo/train.jsonl
uv run python -m src.train --config configs/dft.toml --train-data data/processed/demo/train.jsonl
uv run python -m unittest discover -s tests -v
```

数据脚本产出 `train.jsonl`、`validation.jsonl` 和 `audit.json`；训练脚本输出每步的损失、参与监督的 Token 数与梯度范数。示例用于验证数据、掩码和反向传播，不能用来比较实际模型准确率；SFT 和 DFT 的损失数值尺度也不相同。

## 接入真实实验

1. 固定 [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) 和 [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 的版本；先准备 100–500 条训练样本，完成[数据审计](docs/data-audit.md)。
2. 接入模型、Tokenizer、LoRA 训练和统一的答案评估，先做 Base 评测，再比较同数据下的 SFT 和 DFT。
3. 依据官方论文和代码核对并实现 IDFT 与 Hinted Decoding，补齐 A–E 组并记录成本、数据覆盖率和多种子结果。
