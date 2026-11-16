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

本地对照在开始评分前固定以下选择：官方 [Base 权重固定版本](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/model.safetensors)、vLLM 单卡 BF16、[Qwen 公开 CoT 提示](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/utils.py#L141-L147)与[停止符](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/math_eval.py#L232-L275)。用原生 4096 总上下文减去各题提示的 token 数作为生成上限。显式关闭 top-k / top-p 截断，给每次作答记录不同的确定性种子。答案提取只看最后一个完整答案框，数值先精确比较，符号比较使用固定版本的 [Math-Verify 0.9.0](https://pypi.org/project/math-verify/0.9.0/)；无法提取或解析的作答计为零分并单列。以上未公开的选项是本次复跑的约定，分数可能不同于原表。

测试文件是现有 HuggingFaceH4/MATH-500 固定版本 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` 的 `id/problem/answer` 投影，SHA-256 为 `0baf4950c69ab971f1b45437b965927e5a2a8fda0d2c884c408c9b9c087fca84`。生成只读取题面，判分时才附参考答案。每轮 500 道，每题一次作答，共 8,000 次；先独立评分再取平均，不做投票或择优。MATH-500 不参与选择解码或训练参数。

在解包目录中，用 `bash scripts/run_idft_base.sh setup-score` 创建独立评分环境，再用 `bash scripts/run_idft_base.sh check` 检查索引 6（第七张卡）和 vLLM。脚本默认试用 `nanovllm` 的 Python；如果只有 nano-vllm 而没有 `vllm`，在独立环境按 [vLLM 官方安装文档](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html) 匹配服务器驱动，并用 `DISTALIGN_VLLM_PY` 指向该环境的 Python。评分环境不改训练环境里的 PyTorch 或 SymPy。

`bash scripts/run_idft_base.sh smoke` 用三道手写题各采样两次。流程通过后运行 `bash scripts/run_idft_base.sh math500`；正式输出在 `results/math500/`，含配置、逐轮作答、逐题判分和 `metrics.json`。中断后在同一解包目录重跑同一命令，完整轮次不会重跑。最终用 `mean_accuracy` 和论文的 43.25% 比较；差异需连同本页列出的假设一起报告。

Base 跑通后，可继续复跑 SFT 和 DFT：先固定一份更大、审计过的 Numina 训练子集和独立开发集，给各方法相同的学习率选择预算。用本项目 250 步 LoRA 得到的成绩属于已有小试跑。如果原作者没有公开 1.5B 细节，后续结果应标为公开条件下的独立对照。
