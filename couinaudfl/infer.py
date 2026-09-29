"""Inference: 96×256×256 sliding window at native 512 resolution; z is zero-padded to a multiple of 32 and cropped back."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from .data import normalize, PATCH


def _ceil32(n: int) -> int: return int(np.ceil(n / 32) * 32)


@torch.no_grad()
def predict_volume(model: torch.nn.Module, image_u8: np.ndarray, device, patch=PATCH, overlap: float = 0.5,
                   amp: bool = True, sw_batch: int = 1) -> tuple[np.ndarray, torch.Tensor]:
    """image_u8 (D,H,W) → (label map uint8 (D,H,W), softmax probs (C,D,H,W) on CPU float16)."""
    model.eval()
    D, H, W = image_u8.shape; Dp = max(_ceil32(D), patch[0])
    x = torch.from_numpy(normalize(image_u8))[None, None].to(device)
    if Dp != D: x = F.pad(x, (0, 0, 0, 0, 0, Dp - D))

    def _net(t):
        o = model(t); return o[0] if isinstance(o, (list, tuple)) else o
    with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
        logits = sliding_window_inference(x, roi_size=patch, sw_batch_size=sw_batch, predictor=_net, overlap=overlap, mode="gaussian")
    probs = logits[0, :, :D].float().softmax(0)
    return probs.argmax(0).to(torch.uint8).cpu().numpy(), probs.half().cpu()


FLIPS = [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)]


def _apply_flip(v: np.ndarray, f) -> np.ndarray:
    sl = tuple(slice(None, None, -1) if x else slice(None) for x in f)
    return np.ascontiguousarray(v[sl])


@torch.no_grad()
def detect_orientation(model: torch.nn.Module, image_u8: np.ndarray, device, patch=PATCH, amp: bool = True):
    """For each of the 8 axis-flip hypotheses, run one forward pass on the central patch and pick the hypothesis with the highest foreground confidence.
    Returns (flip tuple, per-hypothesis score dict). Correctly oriented input yields (False, False, False)."""
    model.eval()
    D, H, W = image_u8.shape
    z0 = max((D - patch[0]) // 2, 0); y0 = max((H - patch[1]) // 2, 0); x0 = max((W - patch[2]) // 2, 0)
    scores = {}
    for f in FLIPS:
        im = _apply_flip(image_u8, f)[z0:z0 + patch[0], y0:y0 + patch[1], x0:x0 + patch[2]]
        if im.shape != tuple(patch):
            pad = np.zeros(patch, im.dtype); pad[:im.shape[0], :im.shape[1], :im.shape[2]] = im; im = pad
        x = torch.from_numpy(normalize(im))[None, None].to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            o = model(x); o = o[0] if isinstance(o, (list, tuple)) else o
        p = o.float().softmax(1)[0]
        fg = 1.0 - p[0]
        scores[f] = float((fg * (fg > 0.5)).sum().item())   # confidence-weighted foreground mass
    best = max(scores, key=scores.get)
    return best, scores


@torch.no_grad()
def predict_volume_auto_orient(model: torch.nn.Module, image_u8: np.ndarray, device, patch=PATCH, overlap: float = 0.5,
                               amp: bool = True, sw_batch: int = 1):
    """For inputs of unknown orientation: detect orientation → sliding-window inference in canonical orientation → map labels back to the original orientation.
    Returns (label map, probs (canonical orientation), flip tuple)"""
    f, _ = detect_orientation(model, image_u8, device, patch, amp)
    im = _apply_flip(image_u8, f)
    pred, probs = predict_volume(model, im, device, patch, overlap, amp, sw_batch)
    return _apply_flip(pred, f), probs, f
