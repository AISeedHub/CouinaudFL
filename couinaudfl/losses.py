"""Losses and metrics: 10-class softmax (background + 9 segments); 8-segment labels are partially supervised by summing 4a/4b probabilities.

merge_4ab(logits→probs): merges 4a (4) + 4b (5) of the 10-class probabilities into 9-class probabilities.
In 8-segment cases the label is stored in ch4 (= all of S4) by the preprocessing convention, so labels are mapped to 9-class indices
[0,1,2,3, 4(S4), 5(S5), 6(S6), 7(S7), 8(S8)] and CE+Dice is computed.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

NC = 10
# 10-class labels → 8-segment (9-class) labels: 4a(4),4b(5)→4, S5(6)→5, S6(7)→6, S7(8)→7, S8(9)→8
_MAP10TO9 = torch.tensor([0, 1, 2, 3, 4, 4, 5, 6, 7, 8])


def probs_9(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(1)
    return torch.cat([p[:, :4], p[:, 4:6].sum(1, keepdim=True), p[:, 6:]], dim=1)


def soft_dice(probs: torch.Tensor, onehot: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dims = (0, 2, 3, 4)
    inter = (probs * onehot).sum(dims); denom = probs.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0                      # exclude classes absent from the batch
    d = (2 * inter + eps) / (denom + eps)
    return 1 - d[1:][present[1:]].mean() if present[1:].any() else probs.sum() * 0


def seg_loss(logits: torch.Tensor, label: torch.Tensor, seg8: torch.Tensor, w_ce: float = 1.0, w_dice: float = 1.0):
    """logits (B,10,D,H,W), label (B,D,H,W) long, seg8 (B,) bool ; 9- or 10-class supervision per sample."""
    total = logits.new_zeros(()); n = 0
    for b in range(logits.shape[0]):
        lg, lb = logits[b:b + 1], label[b:b + 1]
        if bool(seg8[b]):
            p = probs_9(lg); lb9 = _MAP10TO9.to(lb.device)[lb]
            ce = F.nll_loss(torch.log(p.clamp_min(1e-7)), lb9)
            oh = F.one_hot(lb9, 9).permute(0, 4, 1, 2, 3).float()
        else:
            p = lg.softmax(1); ce = F.cross_entropy(lg, lb)
            oh = F.one_hot(lb, NC).permute(0, 4, 1, 2, 3).float()
        total = total + w_ce * ce + w_dice * soft_dice(p, oh); n += 1
    return total / max(n, 1)


@torch.no_grad()
def dice_per_class(pred: torch.Tensor, label: torch.Tensor, nc: int = NC) -> torch.Tensor:
    """pred/label (D,H,W) integer label maps → (nc,) Dice; nan for classes absent from both GT and prediction."""
    out = torch.full((nc,), float("nan"))
    for c in range(nc):
        p = pred == c; g = label == c; s = p.sum() + g.sum()
        if g.sum() > 0 or p.sum() > 0: out[c] = (2 * (p & g).sum() / s).float()
    return out


def merge_pred_9(pred: torch.Tensor) -> torch.Tensor:
    """10-class predicted label map → 9-class label map for 8-segment evaluation (4a/4b → 4)."""
    return _MAP10TO9.to(pred.device)[pred.long()]
