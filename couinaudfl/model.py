"""Model: same architecture as TotalSegmentator (nnU-Net v2, PlainConvUNet 3d_fullres) with a 10-class Couinaud head.

TS task 291(total, CT) / 730(total_mr) plans.json:
  PlainConvUNet, base 32, 6 stages, features [32,64,128,256,320,320], conv 2/stage,
  kernel 3, strides [1,2,2,2,2,2] (isotropic), InstanceNorm3d(affine), LeakyReLU, deep supervision.
Weights are transferred from 'network_weights' in the TS checkpoint (checkpoint_final.pth), excluding only the segmentation head (decoder.seg_layers.*).
Our data are anisotropic (4 mm in z, ~0.7 mm in-plane); encoder weights are kept as is and the mismatch is absorbed by fine-tuning.
"""
from __future__ import annotations
import os, glob
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

NUM_CLASSES = 10
TS_FEATURES = [32, 64, 128, 256, 320, 320]
TS_STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


def build_model(num_classes: int = NUM_CLASSES, deep_supervision: bool = True, in_channels: int = 1) -> PlainConvUNet:
    return PlainConvUNet(
        input_channels=in_channels, n_stages=6, features_per_stage=TS_FEATURES, conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6, strides=TS_STRIDES, n_conv_per_stage=[2] * 6, num_classes=num_classes,
        n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=deep_supervision)


def find_ts_checkpoint(root: str) -> str:
    c = sorted(glob.glob(os.path.join(root, "**", "checkpoint_final.pth"), recursive=True))
    if not c: raise FileNotFoundError(f"checkpoint_final.pth not under {root}")
    return c[0]


def load_ts_init(model: nn.Module, ckpt_path: str, verbose: bool = True) -> dict:
    """Transfer TS weights, excluding the seg head (different number of classes). Returns a stats dict."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("network_weights", ck)
    own = model.state_dict(); new = {}; skipped = []
    for k, v in sd.items():
        k2 = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        if k2 in own and own[k2].shape == v.shape and not k2.startswith("decoder.seg_layers"):
            new[k2] = v
        else:
            skipped.append(k2)
    missing = [k for k in own if k not in new]
    model.load_state_dict(new, strict=False)
    st = {"transferred": len(new), "model_tensors": len(own), "skipped_from_ckpt": skipped, "not_initialized": missing}
    if verbose:
        print(f"[TS init] {len(new)}/{len(own)} tensors transferred from {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}; "
              f"not initialized: {len(missing)} ({', '.join(missing[:4])}{' …' if len(missing) > 4 else ''})")
    return st


def load_init(model: nn.Module, path: str, verbose: bool = True) -> str:
    """Load initialization weights from either
    (1) a CouinaudFL state_dict (e.g., outputs/pretrain/best.pth), loaded strictly, or
    (2) an nnU-Net/TotalSegmentator checkpoint (a checkpoint_final.pth file or a folder containing one),
        transferred with load_ts_init (the segmentation head is newly initialized).
    Returns "state_dict" or "totalsegmentator"."""
    if os.path.isdir(path):
        load_ts_init(model, find_ts_checkpoint(path), verbose); return "totalsegmentator"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "network_weights" in ck:
        load_ts_init(model, path, verbose); return "totalsegmentator"
    model.load_state_dict(ck.get("model", ck)); return "state_dict"


def count_params(m: nn.Module) -> int: return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "path/to/totalsegmentator/task291"
    m = build_model(); print(f"params {count_params(m)/1e6:.2f}M")
    st = load_ts_init(m, find_ts_checkpoint(root)); print("skipped:", st["skipped_from_ckpt"])
    m = m.cuda(); x = torch.randn(1, 1, 80, 256, 256, device="cuda")
    with torch.no_grad(): y = m(x)
    print("out", [tuple(t.shape) for t in (y if isinstance(y, (list, tuple)) else [y])])
