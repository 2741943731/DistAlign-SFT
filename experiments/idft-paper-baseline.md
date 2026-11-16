# IDFT 论文表 8：Qwen2.5-Math-1.5B 对照

[论文 §4.1](https://arxiv.org/html/2602.12222v2#S4.SS1) 用 Numina-Math 训练 1.5B 和 7B；除学习率外固定训练参数，给各方法同等预算单独选学习率。其评测继承表 1 的温度 0.3 和 16 轮平均；[附录 B.2](https://arxiv.org/html/2602.12222v2#A2.SS2) 说明生成使用 vLLM。下面只对接表 8 的 MATH-500 一列，先测无需训练的 Base。

| 组别 | 论文报告的 MATH-500 准确率 |
| --- | ---: |
| Base Model | 43.25% |
| SFT | 59.60% |
| DFT | 64.88% |
| EAFT | 59.50% |
| IDFT | 68.20% |

论文和所查到的官方仓库没有给出表 8 对应的完整提示词、生成上限、stop、采样种子和判分器，也未找到 1.5B 的训练样本清单、训练预算及最终学习率。[作者训练脚本](https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT/blob/dc9b5df31673dd2ecfebefe8963ec8a322dd9a6a/IDFT/examples/sft/numina_cot/run_qwen2_5_7b_IDFT.sh) 是 7B-Instruct 示例，不能当作这张表的 SFT 配置。论文 §4.2 的 DeepMath 是另一组实验。

本地 `idft-table8-base-plain-v2` 在正式评分前固定以下选择：官方 [Base 权重固定版本](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/model.safetensors)、vLLM 单卡 BF16、项目的 `Question:/Solution:` 纯文本提示。在 [Qwen 公开的停止符](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/math_eval.py#L232-L275)基础上，遇到新题的 `\n\nQuestion:` 即停止，停止字符串不计入答案。正式评测的生成上限为原生 4096 总上下文减去各题提示的 token 数，手写题试跑另限 384 个新 token。显式关闭 top-k / top-p 截断，给每次作答记录不同的确定性种子。答案提取只看最后一个完整答案框，数值先精确比较，符号比较使用固定版本的 [Math-Verify 0.9.0](https://pypi.org/project/math-verify/0.9.0/)；无法提取或解析的作答计为零分并单列。以上未公开的选项是本次复跑的约定，分数可能不同于原表。

测试文件是现有 HuggingFaceH4/MATH-500 固定版本 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` 的 `id/problem/answer` 投影，SHA-256 为 `0baf4950c69ab971f1b45437b965927e5a2a8fda0d2c884c408c9b9c087fca84`。生成只读取题面，判分时才附参考答案。每轮 500 道，每题一次作答，共 8,000 次；先独立评分再取平均，不做投票或择优。MATH-500 不参与选择解码或训练参数。

在新解包目录中，用 `bash scripts/run_idft_base.sh setup-score` 创建独立评分环境；已有固定版本评分环境可通过 `DISTALIGN_SCORE_PY` 直接复用其 Python。`bash scripts/run_idft_base.sh check` 检查索引 6（第七张卡）和 vLLM。脚本默认选择服务器已有的 `shopping-grpo-longhorizon/.venv`，可用 `DISTALIGN_VLLM_PY` 指定其他解释器。

`bash scripts/run_idft_base.sh smoke` 用三道手写题各采样两次。流程通过后运行 `bash scripts/run_idft_base.sh math500`；正式输出在 `results/math500/`，含配置、逐轮作答、逐题判分和 `metrics.json`。中断后在同一解包目录重跑同一命令，完整轮次不会重跑。最终用 `mean_accuracy` 和论文的 43.25% 比较；差异需连同本页列出的假设一起报告。

Base 跑通后，可继续复跑 SFT 和 DFT：先固定一份更大、审计过的 Numina 训练子集和独立开发集，给各方法相同的学习率选择预算。用本项目 250 步 LoRA 得到的成绩属于已有小试跑。如果原作者没有公开 1.5B 细节，后续结果应标为公开条件下的独立对照。

## 首次试跑排查

服务器 vLLM 0.25.1 的 ChatML 试跑有 6 次生成，全部达到约 4,050 个新 token 的上限；没有一次生成答案框。输出反复出现题干、角色文字或新题。日志保存的是 vLLM 的 `CompletionOutput.text`，评分器只能把这些输出记为 `unparsed`。不能用这一轮试跑作为 Base 分数。新版入口会重判两轮手写题原文；每道题至少一次严格答对才允许启动 MATH-500，已有评分文件不能替代这项检查。

v1 的 `qwen25-math-cot` 是 ChatML 指令提示。Qwen 原报告在 [图 6](https://arxiv.org/pdf/2409.12122#page=33) 用带解题示范的纯文本 `Problem:/Solution:` 评测 Base，图 11 的零样本指令提示用于 Instruct；这仍无法证明 IDFT 表 8 的实际提示。`scripts/probe_idft_prompt.sh` 比较原提示与项目早先用过的 `Question:/Solution:` 纯文本提示。只用三道手写题，锁定 checkpoint、引擎、温度 0.3、每题种子、停止条件和 384 token 上限。

实测 ChatML 三道都到 384 token 上限，得 0/3；纯文本在 365、313、67 token 后停止，分别得到 `408`、`5/8` 和 `5`，用同一个判分器得 3/3。该对照支持在现有后端使用纯文本提示。v2 使用新目录和新协议标识；v1 生成不得并入 v2 分数。MATH-500 未参与这项选择。

## Base v2 正式结果

服务器回传的 [metrics](idft-paper-base-v2-result.json) 共 16 轮，每轮 500 道。严格得分为 **3,909 / 8,000 = 48.8625%**，单轮范围 45.0%–51.2%，比论文表 8 的 43.25% 高 5.6125 个百分点。协议相同的后续实验以 48.8625% 为 Base 对照；提示、停止规则和判分差异仍须随结果记录。

2,415 次为可判的错误答案，1,105 次未提取到答案，571 次为解析或比较不支持。三者都保留在分母里。总共 280 次触及生成上限（3.5%），单凭这个汇总不能把未提取答案归因于截断。服务器应保留同目录下的 `run.json`、16 个原始轮次和 `graded.jsonl`。

下一组先固定训练数据和独立开发集，再开始 SFT。[DFT 论文 §4.1.1](https://arxiv.org/html/2508.05629v2#S4.SS1.SSS1) 公开了一套可参考的设置：100,000 条 NuminaMath-CoT，AdamW、学习率 `5e-5`、有效 batch 256、2048 token、余弦衰减和 10% 预热。这套设置来自 DFT 论文，不是 IDFT 表 8 未公开的 1.5B 配置。单卡复跑需要预先写明样本与训练预算；训练中使用独立开发集调参，最终只用冻结的 v2 协议与本轮 Base 比较。
