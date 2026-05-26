#!/usr/bin/env python3
"""
Profile parameters, FLOPs, and training VRAM for the three model architectures.

Usage:
    ./env/bin/python tools/profile_models.py
    ./env/bin/python tools/profile_models.py --models Quad_L2Net_ConfCFS ViTDense
    ./env/bin/python tools/profile_models.py --res 480 640 --batch 8

Outputs per model:
    - Parameter count and checkpoint size (fp32 / bf16)
    - GFLOPs per image at the given resolution (2×MACs convention)
    - Training VRAM estimate: model states + forward-pass activations × 2

VRAM methodology:
    Model states   = params × 16 bytes
                     (fp32 master weights + bf16 weights + bf16 grads + 2×fp32 Adam moments)
    Activations    = sum of all leaf-module output tensors during one forward pass (bf16),
                     measured by forward hooks — a lower bound that excludes autograd-saved
                     tensors for backward. Multiply the raw hook total by ~2 for a rough
                     full-backward estimate; pairs are already accounted for (×2 in total).
"""

import argparse
import os
import sys
import traceback

import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from models.nets.patchnet import Quad_L2Net_ConfCFS
from models.nets.convnextv2 import ConvNeXtV2
from models.nets.vit_dense import ViTDense

MODELS = {
    "Quad_L2Net_ConfCFS": Quad_L2Net_ConfCFS,
    "ConvNeXtV2":         ConvNeXtV2,
    "ViTDense":           ViTDense,
}


# ── formatting ───────────────────────────────────────────────────────────────

def _fmt_params(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    return f"{n / 1_000:.1f} K"


def _fmt_bytes(b):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024


def _fmt_flops(f):
    if f >= 1e12:
        return f"{f / 1e12:.2f} TFLOPs"
    if f >= 1e9:
        return f"{f / 1e9:.2f} GFLOPs"
    return f"{f / 1e6:.1f} MFLOPs"


# ── measurement ───────────────────────────────────────────────────────────────

def count_flops(model: torch.nn.Module, H: int, W: int, device: str) -> int:
    """FLOPs for a single-image forward pass via FlopCounterMode (2×MACs convention)."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    with FlopCounterMode(model, display=False) as ctr:
        model([dummy])
    return ctr.get_total_flops()


def activation_bytes_fwd(model: torch.nn.Module, H: int, W: int,
                          batch_size: int, device: str) -> int:
    """
    Sum of all leaf-module output tensor bytes (bf16) during one forward pass.

    This is a lower bound on true training activation memory: it captures every
    feature map but not the extra tensors PyTorch saves internally for backward
    (e.g. pre-activation inputs, attention matrices). Add ~2× for a rough
    backward estimate.
    """
    total = 0
    hooks = []

    def _hook(*args):
        out = args[2]  # forward hook signature: (module, input, output)
        nonlocal total
        if isinstance(out, torch.Tensor):
            total += out.numel() * 2          # 2 bytes per bf16 element
        elif isinstance(out, (list, tuple)):
            for t in out:
                if isinstance(t, torch.Tensor):
                    total += t.numel() * 2

    for m in model.modules():
        if not list(m.children()):            # leaf modules only
            hooks.append(m.register_forward_hook(_hook))

    model.eval()
    dummy = torch.randn(batch_size, 3, H, W, device=device)
    with torch.no_grad():
        model([dummy])

    for h in hooks:
        h.remove()
    return total


# ── per-model profiling ───────────────────────────────────────────────────────

def profile_model(name: str, cls, H: int, W: int, batch_size: int, device: str) -> dict:
    model = cls().to(device)
    n = sum(p.numel() for p in model.parameters())

    flops     = count_flops(model, H, W, device)
    act_one   = activation_bytes_fwd(model, H, W, batch_size, device)
    # img_a and img_b must both be live in memory during the backward pass
    act_pair  = act_one * 2
    # 16 bytes/param: bf16 weights (2) + bf16 grads (2) + fp32 master (4) + 2×fp32 Adam (8)
    states    = n * 16
    total     = states + act_pair

    return dict(n=n, flops=flops, act_one=act_one,
                act_pair=act_pair, states=states, total=total)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Profile model size, FLOPs, and training VRAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    pa.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS),
        help="Models to profile (default: all three)",
    )
    pa.add_argument(
        "--res", nargs=2, type=int, default=[192, 192], metavar=("H", "W"),
        help="Input resolution in pixels (default: 192 192, the training crop size)",
    )
    pa.add_argument(
        "--batch", type=int, default=16,
        help="Training batch size used for activation estimates (default: 16)",
    )
    pa.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for forward pass (default: cuda if available, else cpu)",
    )
    args = pa.parse_args()

    H, W, B = *args.res, args.batch
    divider = "─" * 64

    print(f"\nInput {H}×{W}  |  batch={B}  |  device={args.device}")
    print("═" * 64)

    for name in args.models:
        print(f"\n{name}")
        print(divider)
        try:
            s = profile_model(name, MODELS[name], H, W, B, args.device)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            continue

        print(f"  Parameters       {_fmt_params(s['n'])}  ({s['n']:,})")
        print(f"  Checkpoint size  {_fmt_bytes(s['n'] * 4)} (fp32)"
              f"  /  {_fmt_bytes(s['n'] * 2)} (bf16)")
        print(f"  GFLOPs/image     {_fmt_flops(s['flops'])}  (2×MACs convention)")
        print()
        print(f"  Training VRAM estimate  [batch={B}, bf16, paired images]")
        print(f"    Model states    {_fmt_bytes(s['states'])}"
              f"  (fp32 master + bf16 wt/grad + 2×fp32 Adam)")
        print(f"    Activations     {_fmt_bytes(s['act_pair'])}"
              f"  (fwd hooks ×2 for img_a+img_b; lower bound)")
        print(f"    {'─'*44}")
        print(f"    Total           {_fmt_bytes(s['total'])}"
              f"  (~2× more for backward autograd saves)")

    print()


if __name__ == "__main__":
    main()
