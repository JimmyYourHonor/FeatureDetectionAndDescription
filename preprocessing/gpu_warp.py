"""Batched GPU geometric warp.

Pairs with the CPU pipeline in `preprocessing.transform_builder.ParametricTransform`,
which emits raw uint8 source images plus 3x3 ``M_a``/``M_b`` matrices that map
*crop pixel coords* to *source pixel coords* (pixel-corner convention with
simple ``out / in`` scaling — same as ``persp_apply``).

For each sample we evaluate ``M`` at the **pixel-center** of every output
pixel ``(j + 0.5, i + 0.5)``, divide by ``W_src`` / ``H_src``, and shift to
``[-1, 1]`` so it can be fed to ``F.grid_sample`` with
``align_corners=False``.

Source images vary in resolution (no padding/stacking on the CPU side), so we
sample one at a time and stack the rendered crops into a batched output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUWarp(nn.Module):
    def __init__(self, crop_size=(192, 192)):
        super().__init__()
        crop_W, crop_H = crop_size
        self.crop_W = int(crop_W)
        self.crop_H = int(crop_H)

        # Constant homogeneous crop grid: pixel-center coords (j+0.5, i+0.5, 1).
        j = torch.arange(self.crop_W, dtype=torch.float32) + 0.5
        i = torch.arange(self.crop_H, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(i, j, indexing='ij')
        ones = torch.ones_like(gx)
        self.register_buffer(
            '_crop_grid_homo',
            torch.stack([gx, gy, ones], dim=-1),  # (crop_H, crop_W, 3)
            persistent=False,
        )

    def forward(self, src_a, src_b, M_a, M_b):
        """Render ``img_a`` and ``img_b`` from raw sources via grid_sample.

        Args:
            src_a: list of length B of ``(3, H_src, W_src)`` uint8 tensors.
            src_b: list of length B of ``(3, H_src, W_src)`` uint8 tensors.
            M_a:   ``(B, 3, 3)`` float32 'crop pixel → src pixel' homographies.
            M_b:   ``(B, 3, 3)`` float32 'crop pixel → src pixel' homographies.

        Returns:
            img_a, img_b: ``(B, 3, crop_H, crop_W)`` uint8 tensors.
        """
        return self._render(src_a, M_a), self._render(src_b, M_b)

    # ── internals ─────────────────────────────────────────────────────────────

    def _render(self, srcs, M):
        return torch.stack([self._warp_one(s, m) for s, m in zip(srcs, M)], dim=0)

    def _warp_one(self, src, M):
        _, H_src, W_src = src.shape
        device = src.device

        grid_homo = self._crop_grid_homo.to(device)             # (cH, cW, 3)
        src_pts = grid_homo @ M.to(device).t()                   # (cH, cW, 3)
        w = src_pts[..., 2:3]
        src_xy = src_pts[..., :2] / w                           # (cH, cW, 2)

        norm_x = 2.0 * src_xy[..., 0] / W_src - 1.0
        norm_y = 2.0 * src_xy[..., 1] / H_src - 1.0
        norm_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, cH, cW, 2)

        src_f = src.float().unsqueeze(0)                         # (1, 3, H, W)
        out = F.grid_sample(
            src_f, norm_grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        return out.squeeze(0).clamp_(0.0, 255.0).to(torch.uint8)
