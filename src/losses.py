"""Teacher-forced next-token objectives for SFT, DFT, and IDFT."""

import math
from typing import Literal

import torch
from torch.nn import functional as F


LOSS_TYPES = ("sft", "dft", "idft")
IDFT_CONFIG = {"beta": 1.0, "gamma_min": 0.1, "gamma_max": 2.0, "probability_floor": 1e-10}


def _idft_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        entropy = torch.special.entr(log_probs.exp()).sum(dim=-1)
        # phi = log p(target) + H. Clamp log(gamma) before exp to avoid overflow.
        log_gamma = (nll - entropy) / IDFT_CONFIG["beta"]
        gamma = log_gamma.clamp(math.log(IDFT_CONFIG["gamma_min"]), math.log(IDFT_CONFIG["gamma_max"])).exp()
    nll = nll.clamp_max(-math.log(IDFT_CONFIG["probability_floor"]))
    weight = (-gamma * nll.detach()).exp()
    return weight * nll


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_type: Literal["sft", "dft", "idft"] = "sft",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Average the loss over supervised next tokens in the current batch.

    ``logits`` has shape ``[batch, length, vocab]``; ``labels`` has shape
    ``[batch, length]`` and contains unshifted long token IDs, with prompt and
    padding labels already set to ``ignore_index``. The first label and last
    logit in each sequence are unused. An empty supervised batch is an error.

    DFT (https://arxiv.org/abs/2508.05629) uses detached p(target). IDFT
    (https://arxiv.org/abs/2602.12222) uses detached p(target)**gamma, with gamma
    computed from target log probability and vocabulary entropy. All methods
    average over the same supervised tokens.
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"unsupported loss_type: {loss_type!r}; expected one of {LOSS_TYPES}")
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
    if loss_type == "idft":
        return _idft_token_loss(token_logits, targets).mean()

    nll = F.cross_entropy(token_logits, targets, ignore_index=ignore_index, reduction="none")
    if loss_type == "dft":
        nll = nll * nll.detach().neg().exp()
    return nll.mean()
