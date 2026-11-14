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

用手写答案练习逐题判分：

```bash
GRADE_DEMO=$(mktemp -d)
uv run python -m src.grade --predictions examples/grading.jsonl --output "$GRADE_DEMO/graded.jsonl"
cat "$GRADE_DEMO/graded.jsonl"
```

判分脚本读取每题的 `id`、纯生成文本 `completion` 与标准 `answer`，保存逐题提取和判分结果。当前原型只比较最后一个 `\boxed{...}` 中的整数、小数和简单分数；无法提取或不支持的答案单独计数，也留在准确率分母中。正式评测前应使用支持符号答案的评分规则，在独立验证数据上验证并冻结规则。

用 Qwen Base 对三道手写题做一次生成与判分试跑。生成需要 Transformers 5；首次加载会下载约 3.1 GB 的模型文件。`--device auto` 优先使用本机可用的 CUDA 或 MPS，否则使用 CPU。

```bash
BASE_DEMO=$(mktemp -d)
uv run --with 'transformers>=5,<6' python -m src.generate \
  --questions examples/evaluation.jsonl --output-dir "$BASE_DEMO/base" \
  --limit 3 --max-new-tokens 128
uv run python -m src.grade --predictions "$BASE_DEMO/base/predictions.jsonl" \
  --output "$BASE_DEMO/base/graded.jsonl"
cat "$BASE_DEMO/base/graded.jsonl"
```

生成器固定 Base 模型版本，只把题目放进提示，只解码新增的 Token。`predictions.jsonl` 保存生成结果，`run.json` 记录模型版本、输入哈希和生成设置。三道题的分数只用于检查流程。

## 接入真实实验

1. 固定 [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) 和 [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 的版本；先准备 100–500 条训练样本，完成[数据审计](docs/data-audit.md)。
2. 接入模型、Tokenizer、LoRA 训练和统一的答案评估，先做 Base 评测，再比较同数据下的 SFT 和 DFT。
3. 依据官方论文和代码核对并实现 IDFT 与 Hinted Decoding，补齐 A–E 组并记录成本、数据覆盖率和多种子结果。
