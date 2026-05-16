"""GPU-side window selection.

Pairs with the lightweight CPU pipeline in
`preprocessing.transform_builder.ParametricTransform`, which now only samples
geometric parameters (scales, tilts) and emits:

    src_a, src_b   — uint8 source-image tensors (variable resolution, lists)
    sa2ia, sb2ib   — (3, 3) src→img homographies
    M_ab           — (3, 3) img_a→img_b homography (analytic; identity in flow mode)
    img_size       — (4,) int [W_a, H_a, W_b, H_b]
    mode           — 0 = analytic (synthetic/still), 1 = flow
    aflow_full     — (2, H_a, W_a) per-pixel img_a→img_b flow (flow mode only;
                     1×1 placeholder in analytic mode)
    mask_full      — (H_a, W_a) uint8 validity mask (flow mode only;
                     1×1 placeholder in analytic mode)

This module replays the trial-loop crop selection that previously ran in
NumPy on CPU, avoiding the full-image `np.gradient` and aflow grid alloc.
For analytic mode the per-point Jacobian and the (c2x, c2y) lookup are
closed-form; for flow mode we sample the stored flow tensor.

It produces the per-item M_a, M_b, aflow (cropped), mask (cropped) tensors
that GPUWarp + the loss expect — i.e. the same contract as the previous
`ParametricTransform` output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


MODE_ANALYTIC = 0
MODE_FLOW = 1


class GPUWindowSelect(nn.Module):
    def __init__(self, crop_size=(192, 192), n_samples=3, max_attempts_per_sample=30,
                 score_target=0.8):
        super().__init__()
        self.crop_W = int(crop_size[0])
        self.crop_H = int(crop_size[1])
        self.n_samples = int(n_samples)
        self.max_attempts = int(max_attempts_per_sample) * self.n_samples
        self.score_target = float(score_target)

    def forward(self, sa2ia, sb2ib, M_ab, img_size, mode, aflow_full, mask_full):
        """
        Args:
            sa2ia:      (B, 3, 3) float32
            sb2ib:      (B, 3, 3) float32
            M_ab:       (B, 3, 3) float32 (only used when mode[i] == 0)
            img_size:   (B, 4) int [W_a, H_a, W_b, H_b]
            mode:       List[int] of length B (0 = analytic, 1 = flow)
            aflow_full: List[(2, H_a, W_a) float32] (real flow only when mode[i] == 1)
            mask_full:  List[(H_a, W_a) uint8] (real mask only when mode[i] == 1)
        """
        device = sa2ia.device
        B = sa2ia.shape[0]
        cW, cH = self.crop_W, self.crop_H

        M_a = torch.empty(B, 3, 3, device=device, dtype=torch.float32)
        M_b = torch.empty(B, 3, 3, device=device, dtype=torch.float32)
        aflow = torch.empty(B, 2, cH, cW, device=device, dtype=torch.float32)
        mask = torch.empty(B, cH, cW, device=device, dtype=torch.uint8)

        for i in range(B):
            W_a, H_a, W_b, H_b = (int(v) for v in img_size[i].tolist())
            af = aflow_full[i].to(device, non_blocking=True) if mode[i] == MODE_FLOW else None
            mk = mask_full[i].to(device, non_blocking=True) if mode[i] == MODE_FLOW else None

            M_a[i], M_b[i], aflow[i], mask[i] = self._select_one(
                sa2ia[i], sb2ib[i], M_ab[i],
                W_a, H_a, W_b, H_b,
                int(mode[i]), af, mk, device,
            )

        return M_a, M_b, aflow, mask

    # ── per-item selection ────────────────────────────────────────────────────

    def _select_one(self, sa2ia, sb2ib, M_ab, W_a, H_a, W_b, H_b, mode, af, mk, device):
        # Sample candidate centers c1 = (c1x, c1y) in img_a.
        if mode == MODE_FLOW:
            valid = mk.flatten().nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() == 0:
                return self._degenerate(device)
            idx = valid[torch.randint(valid.numel(), (self.max_attempts,), device=device)]
            c1y_all = (idx // W_a).to(torch.long)
            c1x_all = (idx % W_a).to(torch.long)
        else:
            c1x_all = (torch.rand(self.max_attempts, device=device) * W_a).to(torch.long).clamp_(0, W_a - 1)
            c1y_all = (torch.rand(self.max_attempts, device=device) * H_a).to(torch.long).clamp_(0, H_a - 1)

        # Compute (c2, sigma) for all candidates in one shot.
        c1x_f = c1x_all.float()
        c1y_f = c1y_all.float()
        if mode == MODE_FLOW:
            flat_idx = c1y_all * W_a + c1x_all
            af_flat = af.reshape(2, -1)
            c2x_all = af_flat[0, flat_idx]
            c2y_all = af_flat[1, flat_idx]
            sigma_all = self._flow_sigma(af, c1x_all, c1y_all, W_a, H_a)
        else:
            c2x_all, c2y_all, sigma_all = self._analytic_lookup(M_ab, c1x_f, c1y_f)

        # Move scalars to host once for the trial loop control flow.
        c1x_h = c1x_all.tolist()
        c1y_h = c1y_all.tolist()
        c2x_h = c2x_all.tolist()
        c2y_h = c2y_all.tolist()
        sigma_h = sigma_all.tolist()

        best_score = -float('inf')
        best = None
        trials = 0
        for k in range(self.max_attempts):
            if trials >= self.n_samples or best_score >= self.score_target:
                break

            sigma = sigma_h[k]
            c2x_int = int(c2x_h[k] + 0.5)
            c2y_int = int(c2y_h[k] + 0.5)
            if not (0 <= c2x_int < W_b and 0 <= c2y_int < H_b):
                continue

            if 0.2 < sigma < 1:
                w1 = self._window(c1x_h[k], c1y_h[k], self.crop_W / sigma, self.crop_H / sigma, W_a, H_a)
                w2 = self._window(c2x_int, c2y_int, self.crop_W, self.crop_H, W_b, H_b)
            elif 1 <= sigma < 5:
                w1 = self._window(c1x_h[k], c1y_h[k], self.crop_W, self.crop_H, W_a, H_a)
                w2 = self._window(c2x_int, c2y_int, self.crop_W * sigma, self.crop_H * sigma, W_b, H_b)
            else:
                continue

            score = self._score(mode, w1, w2, M_ab, af, mk, device)
            trials += 1
            if score > best_score:
                best_score = score
                best = (w1, w2)

        if best is None:
            return self._degenerate(device)

        return self._build_outputs(best[0], best[1], sa2ia, sb2ib, M_ab, mode, af, mk, device)

    # ── geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _analytic_lookup(M, x, y):
        """Apply 3×3 to (x, y) and return (u, v) plus sqrt|det Jacobian|."""
        m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
        m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
        m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
        w = m20 * x + m21 * y + m22
        u = (m00 * x + m01 * y + m02) / w
        v = (m10 * x + m11 * y + m12) / w
        du_dx = (m00 - u * m20) / w
        du_dy = (m01 - u * m21) / w
        dv_dx = (m10 - v * m20) / w
        dv_dy = (m11 - v * m21) / w
        det = du_dx * dv_dy - du_dy * dv_dx
        sigma = det.abs().clamp_min(1e-16).sqrt()
        return u, v, sigma

    @staticmethod
    def _flow_sigma(af, c1x, c1y, W_a, H_a):
        """Approximate sqrt|det J(flow)| via central differences at integer points."""
        x_p = (c1x + 1).clamp_max(W_a - 1)
        x_m = (c1x - 1).clamp_min(0)
        y_p = (c1y + 1).clamp_max(H_a - 1)
        y_m = (c1y - 1).clamp_min(0)
        af_flat = af.reshape(2, -1)
        # gather along flattened (H, W)
        def g(yy, xx):
            return af_flat[:, yy * W_a + xx]
        dx = g(c1y, x_p) - g(c1y, x_m)
        dy = g(y_p, c1x) - g(y_m, c1x)
        nx = (x_p - x_m).clamp_min(1).float()
        ny = (y_p - y_m).clamp_min(1).float()
        du_dx = dx[0] / nx
        dv_dx = dx[1] / nx
        du_dy = dy[0] / ny
        dv_dy = dy[1] / ny
        det = du_dx * dv_dy - du_dy * dv_dx
        return det.abs().clamp_min(1e-16).sqrt()

    @staticmethod
    def _window(cx, cy, sw, sh, W, H):
        """Replay window1's clamping logic. Returns (xa, ya, xb, yb) with x ∈ [xa, xb)."""
        sw_i = int(0.5 + sw)
        sh_i = int(0.5 + sh)
        xa = cx - int(0.5 + sw / 2)
        xb = xa + sw_i
        if xa < 0:
            xa, xb = 0, xb - xa
        if xb > W:
            xa, xb = xa + W - xb, W
        if xa < 0:
            xa, xb = 0, W
        ya = cy - int(0.5 + sh / 2)
        yb = ya + sh_i
        if ya < 0:
            ya, yb = 0, yb - ya
        if yb > H:
            ya, yb = ya + H - yb, H
        if ya < 0:
            ya, yb = 0, H
        return (xa, ya, xb, yb)

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score(self, mode, w1, w2, M_ab, af, mk, device):
        xa, ya, xb, yb = w1
        x2a, y2a, x2b, y2b = w2

        xs = torch.arange(xa, xb, device=device, dtype=torch.float32)
        ys = torch.arange(ya, yb, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')

        if mode == MODE_FLOW:
            x2 = af[0, ya:yb, xa:xb]
            y2 = af[1, ya:yb, xa:xb]
            mask_in = mk[ya:yb, xa:xb].bool()
        else:
            wh = M_ab[2, 0] * gx + M_ab[2, 1] * gy + M_ab[2, 2]
            x2 = (M_ab[0, 0] * gx + M_ab[0, 1] * gy + M_ab[0, 2]) / wh
            y2 = (M_ab[1, 0] * gx + M_ab[1, 1] * gy + M_ab[1, 2]) / wh
            mask_in = torch.ones_like(x2, dtype=torch.bool)

        x2i = x2.to(torch.long)
        y2i = y2.to(torch.long)
        valid = (x2i >= x2a) & (x2i < x2b) & (y2i >= y2a) & (y2i < y2b)
        score1 = (valid & mask_in).float().mean().item()

        valid_flat = valid.flatten()
        if not valid_flat.any():
            return 0.0
        x2v = x2i.flatten()[valid_flat]
        y2v = y2i.flatten()[valid_flat]
        qx = (16 * (x2v - x2a) // max(x2b - x2a, 1)).clamp_(0, 15)
        qy = (16 * (y2v - y2a) // max(y2b - y2a, 1)).clamp_(0, 15)
        idx = qy * 16 + qx
        accu = torch.zeros(256, device=device, dtype=torch.bool)
        accu[idx] = True
        score2 = accu.float().mean().item()

        return min(score1, score2)

    # ── output construction ───────────────────────────────────────────────────

    def _build_outputs(self, w1, w2, sa2ia, sb2ib, M_ab, mode, af, mk, device):
        xa, ya, xb, yb = w1
        x2a, y2a, x2b, y2b = w2
        Wwa, Hwa = xb - xa, yb - ya
        Wwb, Hwb = x2b - x2a, y2b - y2a
        cW, cH = self.crop_W, self.crop_H

        Win_a = torch.tensor(
            [[Wwa / cW, 0.0, float(xa)],
             [0.0, Hwa / cH, float(ya)],
             [0.0, 0.0, 1.0]],
            device=device, dtype=torch.float32,
        )
        Win_b = torch.tensor(
            [[Wwb / cW, 0.0, float(x2a)],
             [0.0, Hwb / cH, float(y2a)],
             [0.0, 0.0, 1.0]],
            device=device, dtype=torch.float32,
        )
        M_a = torch.linalg.solve(sa2ia, Win_a)
        M_b = torch.linalg.solve(sb2ib, Win_b)

        # Build per-crop-pixel img_a coords (pixel-corner convention) then map to img_b.
        j = torch.arange(cW, device=device, dtype=torch.float32)
        i = torch.arange(cH, device=device, dtype=torch.float32)
        gi, gj = torch.meshgrid(i, j, indexing='ij')
        x_a = xa + gj * (Wwa / cW)
        y_a = ya + gi * (Hwa / cH)

        if mode == MODE_FLOW:
            H_full, W_full = af.shape[1], af.shape[2]
            norm_x = 2.0 * (x_a + 0.5) / W_full - 1.0
            norm_y = 2.0 * (y_a + 0.5) / H_full - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
            sampled = F.grid_sample(
                af.unsqueeze(0), grid,
                mode='nearest', padding_mode='zeros', align_corners=False,
            ).squeeze(0)
            x_b, y_b = sampled[0], sampled[1]
            mask_w = F.grid_sample(
                mk.unsqueeze(0).unsqueeze(0).float(), grid,
                mode='nearest', padding_mode='zeros', align_corners=False,
            ).squeeze(0).squeeze(0).to(torch.uint8)
        else:
            wh = M_ab[2, 0] * x_a + M_ab[2, 1] * y_a + M_ab[2, 2]
            x_b = (M_ab[0, 0] * x_a + M_ab[0, 1] * y_a + M_ab[0, 2]) / wh
            y_b = (M_ab[1, 0] * x_a + M_ab[1, 1] * y_a + M_ab[1, 2]) / wh
            mask_w = torch.ones(cH, cW, device=device, dtype=torch.uint8)

        # Express aflow in crop_b pixel coords.
        x_b = (x_b - x2a) * (cW / Wwb)
        y_b = (y_b - y2a) * (cH / Hwb)
        invalid = ~mask_w.bool()
        x_b = torch.where(invalid, torch.full_like(x_b, float('nan')), x_b)
        y_b = torch.where(invalid, torch.full_like(y_b, float('nan')), y_b)

        aflow_crop = torch.stack([x_b, y_b], dim=0)
        return M_a, M_b, aflow_crop, mask_w

    def _degenerate(self, device):
        cW, cH = self.crop_W, self.crop_H
        M_a = torch.eye(3, device=device, dtype=torch.float32)
        M_b = torch.eye(3, device=device, dtype=torch.float32)
        aflow = torch.full((2, cH, cW), float('nan'), device=device, dtype=torch.float32)
        mask = torch.zeros(cH, cW, device=device, dtype=torch.uint8)
        return M_a, M_b, aflow, mask
