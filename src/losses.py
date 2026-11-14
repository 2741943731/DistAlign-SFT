"""Causal language modeling objectives for SFT and Dynamic Fine-Tuning."""

from typing import Literal

import torch
from torch.nn import functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_type: Literal["sft", "dft"] = "sft",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Average the loss over supervised next tokens in the current batch.

    ``logits`` has shape ``[batch, length, vocab]``; ``labels`` has shape
    ``[batch, length]`` and contains unshifted long token IDs, with prompt and
    padding labels already set to ``ignore_index``. The first label and last
    logit in each sequence are unused. An empty supervised batch is an error.

    DFT reweights each token's NLL by its detached model probability
    (Wu et al., https://arxiv.org/abs/2508.05629). The denominator remains the
    number of supervised tokens for both objectives.
    """
    if loss_type not in ("sft", "dft"):
        raise ValueError(f"unsupported loss_type: {loss_type!r}; expected 'sft' or 'dft'")
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.size(-1) == 0:
        raise ValueError("expected logits [batch, length, vocab] and labels [batch, length]")
    if not logits.is_floating_point() or labels.dtype != torch.long:
        raise TypeError("expected floating-point logits and torch.long labels")
    if logits.device != labels.device:
        raise ValueError("logits and labels must be on the same device")

    targets = labels[:, 1:]
    supervised = targets.ne(ignore_index)
    targets = targets[supervised]
    if targets.numel() == 0:
        raise ValueError("no supervised next-token labels after masking and shifting")

    token_logits = logits[:, :-1, :][supervised]
    if token_logits.dtype in (torch.float16, torch.bfloat16):
        token_logits = token_logits.float()
    nll = F.cross_entropy(token_logits, targets, ignore_index=ignore_index, reduction="none")
    if loss_type == "dft":
        nll = nll * nll.detach().neg().exp()
    return nll.mean()
