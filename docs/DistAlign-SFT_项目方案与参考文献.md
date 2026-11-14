# DistAlign-SFT：分布对齐与动态微调的小模型推理后训练

> **项目定位**：独立的后训练算法实验项目；重点研究 SFT 数据分布、Token 级损失和泛化，不以提出新算法为前提。
>
> **状态**：项目方案 / 待实验。本文所有方法对比均是拟开展的实验，**没有虚构的模型成绩或提升幅度**。
>
> **参考论文**：DFT（ICLR 2026）和 Towards On-Policy SFT（ICML 2026）。

## 1. 项目概述

**中文名称**：基于分布感知优化的小模型推理后训练与泛化研究  
**英文名称**：DistAlign-SFT: Distribution-Aligned Fine-Tuning for Efficient LLM Reasoning

### 1.1 背景

标准监督微调（SFT）对教师生成的目标 Token 使用交叉熵学习，但高质量教师推理轨迹可能与较小学生模型原有的生成分布相距甚远。强行模仿分布外的推理方式，未必能获得良好的泛化能力。

- **DFT（Dynamic Fine-Tuning）**：从强化学习的视角重新理解 SFT 更新，通过目标 Token 概率对损失进行动态缩放。
- **IDFT（In-Distribution Fine-Tuning）**：进一步利用模型分布信息，调整微调目标，研究分布匹配对泛化的影响。
- **Hinted Decoding（HD）**：不只改变损失函数，还通过教师答案提示下的解码过程，构建更贴近学生模型分布、同时维持答案正确性的 CoT 数据。

这些方法都已经由原论文作者提出，**本项目不把 DFT、IDFT 或 HD 署名为原创**。本项目拟贡献的是统一复现、严谨的因子消融，以及对数据分布和泛化关系的实证分析。

### 1.2 核心问题

1. **损失层面**：在同一份教师 CoT 数据上，DFT / IDFT 是否稳定优于标准 SFT？
2. **数据层面**：将教师 CoT 转换为更贴近学生分布的轨迹，能否比单纯修改损失函数带来更明显的改善？
3. **交互层面**：经过 HD 后，IDFT 的额外收益会增大、减小，还是消失？
4. **可选的 RL 对照**：在记录并控制生成、训练和 GPU 成本后，分布对齐 SFT 与在线 GRPO 表现如何？**该比较是本项目提出的实验问题，不是两篇论文已有结论。**

### 1.3 与现有 Shopping-GRPO 项目的区别

| 维度 | Shopping-GRPO | DistAlign-SFT |
|---|---|---|
| 核心场景 | 长程购物 Agent、工具调用 | 数学 CoT 推理与泛化 |
| 主要技术 | Teacher 轨迹、Action-only SFT、在线 GRPO、Reward 与评测 | SFT / DFT / IDFT、Hinted Decoding、分布匹配分析 |
| 主要问题 | Agent 能否完成多轮决策任务 | 训练数据与学生分布偏差如何影响微调 |
| 主要能力证明 | RL 工程与环境 Rollout | Token-level Loss、数据对齐、机制分析与对照实验 |

## 2. 技术原理

### 2.1 标准 SFT

给定问题 $x$ 与教师轨迹 $y=(y_1,\ldots,y_T)$，最常见的 Token 级训练目标为：

$$
\mathcal L_{\mathrm{SFT}}=-\frac1T\sum_{t=1}^T\log p_\theta(y_t\mid x,y_{<t}).
$$

所有非 Padding 的目标 Token 都通过 NLL 参与优化。实际实现还需对 Prompt Token、Padding Token 做 Loss Mask。

### 2.2 DFT：概率加权的 SFT

令 $p_t=p_\theta(y_t\mid x,y_{<t})$，典型 DFT 形式是：

$$
\mathcal L_{\mathrm{DFT}}=-\frac1T\sum_t\operatorname{sg}(p_t)\log p_t,
$$

其中 $\operatorname{sg}$ 为停止梯度（`detach`）：权重由当前模型概率决定，但不会沿权重分支反向传播。

**要理解的关键点**：低概率目标 Token 会得到更小的梯度权重；这并不意味着低概率 Token 必然是不可靠数据。应与标准 SFT 比较，并控制 Loss 缩放、学习率等混杂因素。

### 2.3 IDFT：分布感知的目标调整

IDFT 依据 Distribution Discriminant Theory（DDT）引入目标 Token 的相对分布位置等信息，区分单纯低概率与分布不匹配。它不是简单地把 DFT 的概率权重换个常数。

**实施要求**：按官方论文与代码核对 IDFT 的完整公式、权重构造、梯度截断和数值稳定处理；不要只依据简写公式重写。建议记录目标概率、Entropy、权重分位数以及有效梯度统计。

### 2.4 Hinted Decoding：让数据靠近学生分布

输入同一问题与教师正确解答，通过提示条件化的生成过程引导学生产出候选 CoT，并对答案进行验证。得到更贴合学生自身生成习惯的训练轨迹，再用于标准 SFT 或 IDFT。

**重要区别**：HD 是教师辅助、向学生分布靠拢的数据构造方案，**不等同于无教师的严格 On-policy RL**。

**质量控制**：正确答案不保证中间 CoT 每一步都可靠；需要保留筛选失败率、轨迹长度、题目难度、验证策略及原始教师数据的对应关系。

## 3. 推荐实验配置

| 项目 | 首选方案 | 说明 |
|---|---|---|
| 初始模型 | `Qwen/Qwen2.5-Math-1.5B` | 固定所有实验的初始权重；Base 与 Instruct 不得混用 |
| 教师数据 | `AI-MO/NuminaMath-CoT` 的清洗子集 | 初始 5k–10k，可先做 100–500 条通路验证 |
| 验证数据 | 从训练来源中独立划分 | 不使用正式测试集调阈值、学习率、Checkpoint |
| 测试 | MATH-500；另选一个独立数学基准 | GSM8K 可用，但需严格查重/排除同源样本 |
| 训练方式 | 先统一 LoRA | 如需要验证全参结果，再单独开展全参实验 |
| RL 对照 | GRPO，作为扩展 | 复用既有 RL 经验，不是第一阶段必要条件 |

### 数据集污染风险（必须处理）

NuminaMath-CoT 的数据源包含 GSM8K、MATH 等已有公开数学数据，因此**不能直接随机抽取 NuminaMath 子集，同时宣称 GSM8K Test / MATH-500 是完全未见测试**。至少应：

1. 根据 source 元数据过滤 `gsm8k`、`math` 等可能与目标 Benchmark 交叠的来源；
2. 对规范化题面做精确去重与近重复检测，保存查重规则和数量；
3. 检查是否存在相同题目的翻译版本或改写版本；
4. 对仍无法排除的污染风险明确报告，不夸大泛化结论。

## 4. 分阶段执行方案

### Phase 0：准备数据与 Baseline

- [ ] 下载固定模型与 Tokenizer，记录版本 / Commit / 数据哈希。
- [ ] 清洗 NuminaMath 子集、过滤疑似 Benchmark 重合题目。
- [ ] 固定 Chat Template、答案抽取器、解码参数和评估脚本。
- [ ] 对未经训练的 Base 模型进行一次完整评测。
- [ ] 从 100–500 条数据验证 SFT Loss、Mask、日志和导出流程。

**验收**：数据集可复现；基础模型可评估；无 Train/Test 混用；训练结束可重新加载和推理。

### Phase 1：SFT、DFT、IDFT 同数据比较

- [ ] 统一训练框架，提供 `loss_type` 配置切换。
- [ ] 对相同输入用小 Batch 核验 Loss 和梯度方向。
- [ ] 固定数据顺序与随机种子，分别训练标准 SFT、DFT、IDFT。
- [ ] 为不同 Loss 做公平学习率搜索或报告学习率敏感性。
- [ ] 记录准确率、训练损失、Token 权重分布、峰值显存、GPU 时间。

**验收**：三种 Loss 在完全相同的数据上形成可复现对照；确认提升并非仅由不同有效学习率带来。

### Phase 2：Hinted Decoding 数据构建

- [ ] 从同一批训练题构建 HD 候选轨迹；记录教师原始解答与 HD 解答关联。
- [ ] 验证最终答案；保留通过率、回退率、轨迹长度和难度分布。
- [ ] 计算固定初始学生模型在原始 / HD 轨迹上的 Log-Likelihood 等分布匹配指标。
- [ ] 建立等规模数据的 HD + SFT、HD + IDFT 训练组。

**验收**：能证明 HD 数据在被定义的分布指标上与学生更接近（若不成立，应报告失败）；并在等规模、尽可能等 Token 预算下完成公平对照。

### Phase 3：泛化与机制分析

- [ ] 对测试题按难度 / 题型切片；观察收益分布。
- [ ] 对训练轨迹按**冻结初始模型**下的匹配度分组，防止用训练后模型造成循环定义。
- [ ] 对 HD 的通过率与样本选择偏差做分析，避免“仅保留简单题”导致虚假提升。
- [ ] 检验 Token 权重是否过度压制难题关键位置。
- [ ] 对主要结果重复多个随机种子；给出置信区间或方差。

**验收**：除总分外，至少能解释一种明确的有效或失效场景。

### Phase 4（可选）：在线 GRPO 对照

- [ ] 在相同初始模型和独立数据划分下建立 Outcome-Reward GRPO 对照。
- [ ] 可额外研究标准 SFT 与 HD-SFT 作为 RL 初始化的差异，但这属于新增实验。
- [ ] 分别报告 GPU 时间、RL Rollout Token、SFT 训练 Token、成功率 / 准确率。
- [ ] 明确在线 GRPO 与教师辅助 HD 在数据来源上的不同，不宣称严格预算相等，除非已进行充分控制。

**验收**：仅在可比训练预算和评估协议下讨论“谁更高效”；否则将结论限定为特定设置的观察。

## 5. 最小实验矩阵

| 编号 | 训练数据 | 优化方法 | 主要比较 |
|---|---|---|---|
| A | 原始教师 CoT | 标准 SFT | 基线 |
| B | 原始教师 CoT | DFT | 与 A 比，验证 Token 概率加权 |
| C | 原始教师 CoT | IDFT | 与 A、B 比，验证分布感知 Loss |
| D | HD 生成 CoT | 标准 SFT | 与 A 比，验证数据分布对齐 |
| E | HD 生成 CoT | IDFT | 与 C、D 比，验证数据和 Loss 是否互补 |
| F（扩展） | 在线采样 | GRPO | 与 A–E 比，考察在线 RL 与计算成本 |

**建议优先完成 A–E**；如果工作量、显存或经费受限，不必为了简历强行补 F。可附带 `HD + DFT` 以补足完整的 2×3 因子设计。

## 6. 评估指标与控制变量

### 6.1 核心指标

- **准确率**：统一的最终答案抽取和判分；对不同解码设置分别报告。
- **泛化**：独立测试集、不同题目难度和题型的表现；避免把训练集正确率称作泛化。
- **成本**：训练 GPU 小时、峰值显存、数据生成 Token 数、训练 Token 数、推理 Token 数。
- **数据分析**：HD 筛选通过率、原始与 HD 轨迹的 Token Log-Likelihood、长度、难度及去重结果。
- **机制分析**：DFT / IDFT 权重分布、目标 Token 概率、Entropy、梯度范数和学习率敏感性。

### 6.2 关键控制

- 相同模型初始权重、Tokenizer、最大长度和训练样本划分。
- 尽可能匹配训练 Token 预算，并另行报告生成数据成本。
- 不直接拿同一学习率下的 Loss 加权结果断言算法优劣。
- 采用固定、统一的生成温度、长度限制和答案评估规则。
- HD 的训练题尽量与原始数据组一一对应，记录过滤后的题目覆盖率。
- 测试集只用于最终评估，不用来挑选算法超参数。

## 7. 代码结构建议

```text
DistAlign-SFT/
├── README.md
├── configs/
│   ├── model.yaml
│   ├── sft.yaml
│   ├── dft.yaml
│   ├── idft.yaml
│   └── hinted_decoding.yaml
├── data/
│   ├── prepare.py
│   ├── deduplicate.py
│   └── validate.py
├── src/
│   ├── losses.py             # SFT / DFT / IDFT，及梯度测试
│   ├── hinted_decoding.py    # 生成与样本审计
│   ├── train.py              # 统一训练入口
│   └── evaluate.py           # 统一答案抽取和评估
├── scripts/
├── tests/
├── experiments/
│   ├── configs/
│   ├── metrics/
│   └── ablations.md
└── docs/
    ├── data-audit.md
    ├── methods.md
    └── results.md
```

## 8. 项目完成标准与简历呈现

### 最低可交付（不要求 RL）

- [ ] 统一实现并验证 SFT、DFT、IDFT。
- [ ] 可审计的 HD 数据构造和清洗流程。
- [ ] 完成 A–E 至少五组主要对照。
- [ ] 可信的测试集隔离、复现实验配置和评测脚本。
- [ ] 至少一份解释算法有效 / 失效条件的分析报告。

### 可在简历中突出（完成后按实际结果写）

1. **分布感知优化**：复现并统一实现 DFT / IDFT，分析目标 Token 概率、Entropy 与梯度权重的关系。
2. **学生分布对齐的数据构造**：实现 Hinted Decoding、答案验证和数据审计，研究教师 CoT 与学生分布的差异。
3. **严格消融与泛化评估**：在同数据、同模型、可比预算下对照不同训练方法，完成跨难度分析和成本评估。

> 不要预先填写提升百分比。只有具备真实训练日志、测试协议和统计结果后，才能写“提升 X 个百分点”或“节约 X% 成本”。

## 9. 实施风险

| 风险 | 应对方式 |
|---|---|
| HD 生成成本较高 | 先用小规模 Demo 验证，再批量生成；完整记录采样成本 |
| IDFT 实现细节复杂 | 以论文 / 官方代码为准；添加数值与梯度单测 |
| 算法差异只是有效学习率变化 | 对不同方法调优学习率、记录梯度范数 |
| NuminaMath 与 Benchmark 存在污染 | 根据来源 + 题面相似度双重过滤，记录审计报告 |
| HD 数据只保留容易题 | 题目配对、难度分层与样本覆盖率分析 |
| 单次评测波动较大 | 多种子、固定协议、置信区间 |
| GRPO 成本高 | 保持为可选扩展；优先完成 A–E |

---

## 10. 参考文献与官方资源

### 核心论文（必读）

**[1] Zhang et al. (2026). _Towards On-Policy SFT: Distribution Discriminant Theory and its Applications in LLM Training._ ICML 2026.**

- 论文（arXiv）：https://arxiv.org/abs/2602.12222
- 正式发表页面（PMLR）：https://proceedings.mlr.press/v306/zhang26dp.html
- 官方代码（IDFT + Hinted Decoding Demo）：https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT
- **主要用途**：理解 DDT、IDFT、Hinted Decoding 的原理和实现，并参考数据 / Loss 联合消融。官方仓库列出的高吞吐 vLLM 版 HD Demo 仍标记为进行中；复现前请再次检查实际发布状态。

**[2] Wu et al. (2026). _On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification._ ICLR 2026.**

- 论文（arXiv）：https://arxiv.org/abs/2508.05629
- ICLR 正式页面：https://proceedings.iclr.cc/paper_files/paper/2026/hash/2e53c02ea028cbf603f4b6b47fef3d97-Abstract-Conference.html
- 官方代码：https://github.com/yongliang-wu/DFT
- **主要用途**：理解 SFT 与策略梯度的理论联系、DFT 的 Stop-Gradient 权重实现、方法适用范围和局限。

### 扩展方法（做 GRPO 对照时阅读）

**[3] Shao et al. (2024). _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models._**

- https://arxiv.org/abs/2402.03300
- **主要用途**：GRPO 的原始方法、组内优势估计和数学推理 RL 背景。

### 模型、数据、评测与训练框架

**[4] Qwen2.5-Math-1.5B（建议初始学生模型）**  
https://huggingface.co/Qwen/Qwen2.5-Math-1.5B

**[5] NuminaMath-CoT（建议教师 CoT 数据来源）**  
https://huggingface.co/datasets/AI-MO/NuminaMath-CoT

**[6] MATH-500（建议独立测试集）**  
https://huggingface.co/datasets/HuggingFaceH4/MATH-500

**[7] GSM8K（可选独立测试集，务必排查与训练数据重合）**  
https://huggingface.co/datasets/openai/gsm8k

**[8] Hugging Face TRL（可作为 SFT / DFT 训练框架参考）**  
https://huggingface.co/docs/trl/index

**[9] verl（可选 GRPO 扩展使用）**  
https://github.com/verl-project/verl

---

## 11. 建议阅读顺序

1. **DFT 论文与代码**：先能完整解释标准 SFT、DFT 的梯度差异与 `detach`。
2. **Towards On-Policy SFT 方法部分**：理解 DDT、IDFT，以及“低概率 ≠ 分布外”的区别。
3. **Hinted Decoding Demo**：搞清楚什么是教师引导数据构建、如何保留正确性。
4. **实验部分**：观察不同模型、数据条件下方法并不总是单调提升；识别失败案例。
5. **DeepSeekMath / GRPO**：仅当准备做 Phase 4 时补充阅读。

**最终建议：先完成 A–E 的统一、可复现对照，把真正的研究结论建立起来，再决定是否补做 GRPO。**
