# DistAlign-SFT

研究小模型的数学推理后训练：在同一份教师 CoT 上比较 SFT、DFT、IDFT，再通过 Hinted Decoding 调整数据分布，检查数据与损失的交互如何影响泛化。[原始项目方案与参考文献](docs/DistAlign-SFT_项目方案与参考文献.md)保留了完整实验设计。

目前已包含 JSONL 清洗、SFT / DFT / IDFT 损失、生成与判分，以及使用本地 Qwen 权重的单卡 LoRA 试跑入口。[实验登记表](experiments/ablations.md)记录 A–E 组的进度；试跑用于验证流程，正式对照尚未完成。

## 本地验证

使用 Python 3.11 或更新版本及 PyTorch。依赖安装后，示例不需要下载数据或模型。以下命令在项目根目录运行：

```bash
uv sync --python 3.12
uv run python -m data.prepare --input examples/teacher.jsonl \
  --held-out examples/heldout.jsonl --output-dir data/processed/demo
uv run python -m src.train --config configs/sft.toml --train-data data/processed/demo/train.jsonl
uv run python -m src.train --config configs/dft.toml --train-data data/processed/demo/train.jsonl
uv run python -m src.train --config configs/idft.toml --train-data data/processed/demo/train.jsonl
uv run python -m unittest discover -s tests -v
```

数据脚本产出 `train.jsonl`、`validation.jsonl` 和 `audit.json`；训练脚本输出每步的损失、参与监督的 Token 数与梯度范数。示例用于验证数据、掩码和反向传播。三种损失的权重不同，损失数值不能直接用来比较模型效果。

用手写答案练习逐题判分：

```bash
GRADE_DEMO=$(mktemp -d)
uv run python -m src.grade --predictions examples/grading.jsonl --output "$GRADE_DEMO/graded.jsonl"
cat "$GRADE_DEMO/graded.jsonl"
```

判分脚本读取每题的 `id`、纯生成文本 `completion` 与标准 `answer`，保存逐题提取和判分结果。默认规则比较最后一个 `\boxed{...}` 中的整数、小数和简单分数；无法提取或不支持的答案单独计数，也留在准确率分母中。

用手写例子检查符号判分。`--scorer symbolic` 仍提取最后一个答案框，用精确算术比较根式、含 π 的实数表达式、LaTeX 分式和独立的百分数。例如 `\sqrt{8}` 和 `2\sqrt{2}` 判为等价。额外依赖是固定版本的 Lark 和 SymPy；评分汇总标记为 `symbolic-v1`。

```bash
SYMBOLIC_DEMO=$(mktemp -d)
uv run --extra symbolic-eval python -m src.grade --scorer symbolic \
  --predictions examples/grading-symbolic.jsonl --output "$SYMBOLIC_DEMO/graded.jsonl"
```

该试验规则只支持有界的实数表达式。方程、集合、带单位的答案和残缺式仍记为 `unsupported`；未提取到完整答案框的仍记为 `unparsed`。旧命令继续使用原来的数值规则。正式评测前，先在独立开发题上核对答案类型的覆盖率和判分准确性，再固定评分规则。

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

## 服务器上的 LoRA 试跑

使用已经能运行生成脚本的 PyTorch / Transformers 5 环境。训练与加载 LoRA 还需要 [PEFT](https://huggingface.co/docs/peft/en/quicktour)。先检查 `python -c 'import peft; print(peft.__version__)'`；如果缺少 PEFT，可用 `python -m pip install --dry-run 'peft==0.21.3'` 检查安装计划，确认不会替换当前 PyTorch 和 Transformers 后再去掉 `--dry-run` 安装。

先用 8 道训练题做一次保存和重载检查。把下方路径替换为服务器上的实际路径，运行前进入项目目录并选择已分配的 GPU：

```bash
MODEL=/path/to/Qwen2.5-Math-1.5B
TRAIN_DATA=data/processed/train.jsonl
QUESTIONS=data/processed/lesson4/numeric-validation.jsonl
mkdir -p runs
PILOT=$(mktemp -d "$PWD/runs/lora-pilot.XXXXXX")

python -B -m src.hf_train \
  --model "$MODEL" --train-data "$TRAIN_DATA" --output-dir "$PILOT/sft" \
  --loss-type sft --device cuda --limit 8 --epochs 1 --batch-size 1 --max-length 1024 &&
python -B -m src.generate \
  --model "$MODEL" --adapter "$PILOT/sft/adapter" --local-files-only --device cuda \
  --questions "$QUESTIONS" --output-dir "$PILOT/eval" --limit 15 --max-new-tokens 1024 &&
python -B -m src.grade --predictions "$PILOT/eval/predictions.jsonl" \
  --output "$PILOT/eval/graded.jsonl"
```

训练只更新 `q_proj` 和 `v_proj` 上的 LoRA 参数，并使用与生成一致的题目模板。题目和补齐部分不计入损失；超过 `--max-length` 的完整解答会跳过，数量写入训练的 `run.json`。`sft/steps.jsonl` 记录每步损失和梯度，`sft/adapter/` 保存小型 adapter。重载后用同一批开发题、同一判分规则和 1024 个新 Token 预算，便于和 Base 试跑对照。确认流程正常后，再用同一份训练数据和设置，把 `--loss-type` 改为 `dft` 并使用新的输出目录。

C 组使用 `--loss-type idft`。先做同样的 8 条试跑，再扩大训练。见 [IDFT 损失与实现参数](docs/idft.md)。

## 接入真实实验

1. 固定 [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) 和 [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 的版本；先准备 100–500 条训练样本，完成[数据审计](docs/data-audit.md)。
2. 接入模型、Tokenizer、LoRA 训练和统一的答案评估，先做 Base 评测，再比较同数据下的 SFT 和 DFT。
3. 依据官方论文和代码核对并实现 IDFT 与 Hinted Decoding，补齐 A–E 组并记录成本、数据覆盖率和多种子结果。
