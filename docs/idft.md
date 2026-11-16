# IDFT 损失

在同一个教师解答前缀下，记模型对下一个目标 token 的概率为 $p$，对整个词表的预测分布为 $P$。SFT 的 token 权重为 1，DFT 为 $\operatorname{stopgrad}(p)$。IDFT 先计算词表熵 $H(P)=-\sum_v P(v)\log P(v)$，再算中心化对数概率 $\phi=\log p+H(P)$。较低的 $p$ 在一个本来就很分散的预测中并不罕见，因此不能只看 $p$ 来判断目标 token 是否偏离当前模型。

本项目按[论文第 3.1 节](https://arxiv.org/html/2602.12222v2#S3.SS1)和[作者发布的训练器](https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT/blob/main/IDFT/verl/trainer/fsdp_polylog_trainer.py#L394-L442)实现每个受监督 token 的损失：

$$
\gamma=\mathrm{clip}(\exp(-\phi / \beta),0.1,2.0),\qquad
\tilde p=\max(p,10^{-10}),\qquad
\ell=-\operatorname{stopgrad}(\tilde p^\gamma)\log\tilde p.
$$

例如目标 token 的概率同为 0.25。四个 token 概率都为 0.25 时，$H(P)=-\log 0.25$，$\gamma=1$，权重为 0.25。分布为 `(0.25, 0.73, 0.01, 0.01)` 时熵更低，$\gamma$ 截到 2，权重变成 0.0625。DFT 在两处都会给 0.25。`stopgrad` 让反向传播只使用算出的权重，梯度不经过权重的计算过程。

`--loss-type idft` 使用作者示例的 $\beta=1$、指数范围 `[0.1, 2.0]`、无指数预热和过渡。实现用 `log_softmax` 提取目标 log 概率，并在指数计算前截断 `log(gamma)`。熵按 $0\log 0=0$ 计算；FP16 / BF16 logits 先转成 FP32。和 SFT、DFT 一样，只在解答及 EOS 的位置取平均；训练仍使用项目的单卡 LoRA 设置。`run.json` 的 `idft` 字段记录上述参数。上游对照的训练器文件内容标识为 `1b93848421d3d6525b2dfef712298af16922c748`。
