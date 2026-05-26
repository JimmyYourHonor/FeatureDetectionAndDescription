"""GPU-side window selection (Phase B: vectorized).

Phase A did per-item GPU computation with .tolist() / .item() syncs across
K candidates inside the trial loop. Phase B scores all K candidates in one
tensor pass and batches every analytic item across the batch axis, so the
outer Python loop runs only for flow items (variable-resolution flow tensors
don't pad cleanly).

Pairs with the lightweight CPU pipeline in
`preprocessing.transform_builder.ParametricTransform`, which now only samples
geometric parameters and emits:

    src_a, src_b   — uint8 source-image tensors (variable resolution, lists)
    sa2ia, sb2ib   — (3, 3) src→img homographies
    M_ab           — (3, 3) img_a→img_b homography (analytic; identity in flow mode)
    img_size       — (4,) int [W_a, H_a, W_b, H_b]
    mode           — 0 = analytic (synthetic/still), 1 = flow
    aflow_full     — (2, H_a, W_a) per-pixel img_a→img_b flow (flow mode only)
    mask_full      — (H_a, W_a) uint8 validity mask (flow mode only)

Same external contract as Phase A — see `forward`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


MODE_ANALYTIC = 0
MODE_FLOW = 1


class GPUWindowSelect(nn.Module):
    # Samples-per-side inside each candidate window when scoring. 32×32 = 1024
    # samples per candidate is plenty for ranking 16×16-bucket occupancy.
    SCORE_M = 32

    def __init__(self, crop_size=(192, 192), n_samples=3, max_attempts_per_sample=30,
                 score_target=0.8):
        super().__init__()
        self.crop_W = int(crop_size[0])
        self.crop_H = int(crop_size[1])
        self.n_samples = int(n_samples)
        # Phase B scores all candidates in parallel and picks the best, so
        # n_samples and score_target are subsumed by max_attempts.
        self.max_attempts = int(max_attempts_per_sample) * self.n_samples
        self.score_target = float(score_target)

    # ──────────────────────── public API ──────────────────────────────────────

    def forward(self, sa2ia, sb2ib, M_ab, img_size, mode, aflow_full, mask_full):
        """
        Args:
            sa2ia:      (B, 3, 3) float32
            sb2ib:      (B, 3, 3) float32
            M_ab:       (B, 3, 3) float32 (only used when mode[i] == 0)
            img_size:   (B, 4) int [W_a, H_a, W_b, H_b]
            mode:       List[int] of length B (0 = analytic, 1 = flow)
            aflow_full: List[(2, H_a, W_a) float32] (real flow only when mode[i] == 1)
            mask_full:  List[(H_a, W_a) uint8]      (real mask only when mode[i] == 1)
        """
        device = sa2ia.device
        B = sa2ia.shape[0]
        cW, cH = self.crop_W, self.crop_H

        M_a = torch.empty(B, 3, 3, device=device, dtype=torch.float32)
        M_b = torch.empty(B, 3, 3, device=device, dtype=torch.float32)
        aflow = torch.empty(B, 2, cH, cW, device=device, dtype=torch.float32)
        mask = torch.empty(B, cH, cW, device=device, dtype=torch.uint8)

        analytic_items = [i for i, m in enumerate(mode) if m == MODE_ANALYTIC]
        flow_items = [i for i, m in enumerate(mode) if m == MODE_FLOW]

        if analytic_items:
            idx = torch.tensor(analytic_items, device=device, dtype=torch.long)
            ma_p, mb_p, af_p, mk_p = self._select_analytic_batched(
                sa2ia.index_select(0, idx),
                sb2ib.index_select(0, idx),
                M_ab.index_select(0, idx),
                img_size.index_select(0, idx),
                device,
            )
            M_a.index_copy_(0, idx, ma_p)
            M_b.index_copy_(0, idx, mb_p)
            aflow.index_copy_(0, idx, af_p)
            mask.index_copy_(0, idx, mk_p)

        for i in flow_items:
            W_a, H_a, W_b, H_b = (int(v) for v in img_size[i].tolist())
            af = aflow_full[i].to(device, non_blocking=True)
            mk = mask_full[i].to(device, non_blocking=True)
            M_a[i], M_b[i], aflow[i], mask[i] = self._select_one_flow(
                sa2ia[i], sb2ib[i], W_a, H_a, W_b, H_b, af, mk, device,
            )
            del af, mk

        return M_a, M_b, aflow, mask

    # ────────────────── analytic items (batched) ──────────────────────────────

    def _select_analytic_batched(self, sa2ia, sb2ib, M_ab, img_size, device):
        """Score all K candidates for all N analytic items in one tensor pass."""
        N = sa2ia.shape[0]
        K = self.max_attempts
        cW, cH = self.crop_W, self.crop_H

        W_a = img_size[:, 0].to(device=device, dtype=torch.long)
        H_a = img_size[:, 1].to(device=device, dtype=torch.long)
        W_b = img_size[:, 2].to(device=device, dtype=torch.long)
        H_b = img_size[:, 3].to(device=device, dtype=torch.long)

        # Sample K candidate centers per item.
        c1x = (torch.rand(N, K, device=device) * W_a[:, None].float()).long()
        c1y = (torch.rand(N, K, device=device) * H_a[:, None].float()).long()
        c1x = torch.minimum(c1x, W_a[:, None] - 1).clamp_min(0)
        c1y = torch.minimum(c1y, H_a[:, None] - 1).clamp_min(0)

        c2x, c2y, sigma = self._analytic_lookup_batched(M_ab, c1x.float(), c1y.float())

        sw1, sh1, sw2, sh2 = self._window_sizes(sigma)
        scale_ok = ((sigma > 0.2) & (sigma < 1)) | ((sigma >= 1) & (sigma < 5))
        c2x_int = (c2x + 0.5).long()
        c2y_int = (c2y + 0.5).long()
        in_bounds = ((c2x_int >= 0) & (c2x_int < W_b[:, None]) &
                     (c2y_int >= 0) & (c2y_int < H_b[:, None]))
        valid_cand = scale_ok & in_bounds

        xa, ya, xb, yb = self._window_vec(c1x, c1y, sw1, sh1,
                                          W_a[:, None], H_a[:, None])
        x2a, y2a, x2b, y2b = self._window_vec(c2x_int, c2y_int, sw2, sh2,
                                              W_b[:, None], H_b[:, None])

        scores = self._score_analytic_vec(
            xa, ya, xb, yb, x2a, y2a, x2b, y2b, M_ab, device,
        )
        scores = torch.where(valid_cand, scores, torch.full_like(scores, -1.0))

        best_k = scores.argmax(dim=1)
        idx_n = torch.arange(N, device=device)
        xa_b = xa[idx_n, best_k]
        ya_b = ya[idx_n, best_k]
        xb_b = xb[idx_n, best_k]
        yb_b = yb[idx_n, best_k]
        x2a_b = x2a[idx_n, best_k]
        y2a_b = y2a[idx_n, best_k]
        x2b_b = x2b[idx_n, best_k]
        y2b_b = y2b[idx_n, best_k]
        good = scores[idx_n, best_k] >= 0

        return self._build_outputs_analytic_batched(
            xa_b, ya_b, xb_b, yb_b, x2a_b, y2a_b, x2b_b, y2b_b,
            sa2ia, sb2ib, M_ab, good, device,
        )

    # ────────────────── flow items (per-item; inner vectorized) ───────────────

    def _select_one_flow(self, sa2ia, sb2ib, W_a, H_a, W_b, H_b, af, mk, device):
        K = self.max_attempts

        valid_pixels = mk.flatten().nonzero(as_tuple=False).squeeze(-1)
        if valid_pixels.numel() == 0:
            return self._degenerate(device)

        idx = valid_pixels[torch.randint(valid_pixels.numel(), (K,), device=device)]
        c1y_flat = (idx // W_a).long()
        c1x_flat = (idx % W_a).long()

        flat = c1y_flat * W_a + c1x_flat
        af_flat = af.reshape(2, -1)
        c2x_flat = af_flat[0, flat]
        c2y_flat = af_flat[1, flat]
        sigma_flat = self._flow_sigma(af, c1x_flat, c1y_flat, W_a, H_a)

        # Lift to (1, K) so we share the vectorized helpers with the analytic path.
        c1x = c1x_flat[None]; c1y = c1y_flat[None]
        c2x = c2x_flat[None]; c2y = c2y_flat[None]; sigma = sigma_flat[None]

        sw1, sh1, sw2, sh2 = self._window_sizes(sigma)
        scale_ok = ((sigma > 0.2) & (sigma < 1)) | ((sigma >= 1) & (sigma < 5))
        c2x_int = (c2x + 0.5).long()
        c2y_int = (c2y + 0.5).long()
        Wb_t = torch.tensor([[W_b]], device=device, dtype=torch.long)
        Hb_t = torch.tensor([[H_b]], device=device, dtype=torch.long)
        Wa_t = torch.tensor([[W_a]], device=device, dtype=torch.long)
        Ha_t = torch.tensor([[H_a]], device=device, dtype=torch.long)
        in_bounds = (c2x_int >= 0) & (c2x_int < Wb_t) & (c2y_int >= 0) & (c2y_int < Hb_t)
        valid_cand = scale_ok & in_bounds

        xa, ya, xb, yb = self._window_vec(c1x, c1y, sw1, sh1, Wa_t, Ha_t)
        x2a, y2a, x2b, y2b = self._window_vec(c2x_int, c2y_int, sw2, sh2, Wb_t, Hb_t)

        scores = self._score_flow_vec(xa, ya, xb, yb, x2a, y2a, x2b, y2b, af, mk, device)
        scores = torch.where(valid_cand, scores, torch.full_like(scores, -1.0))

        best_k = scores.argmax(dim=1)[0]
        if scores[0, best_k].item() < 0:
            return self._degenerate(device)

        return self._build_outputs_flow(
            xa[0, best_k], ya[0, best_k], xb[0, best_k], yb[0, best_k],
            x2a[0, best_k], y2a[0, best_k], x2b[0, best_k], y2b[0, best_k],
            sa2ia, sb2ib, af, mk, device,
        )

    # ────────────────── geometry helpers ──────────────────────────────────────

    @staticmethod
    def _analytic_lookup_batched(M, x, y):
        """Apply (N, 3, 3) homography to (N, K) (x, y). Returns (N, K) u, v, sigma."""
        m00 = M[:, 0, 0:1]; m01 = M[:, 0, 1:2]; m02 = M[:, 0, 2:3]
        m10 = M[:, 1, 0:1]; m11 = M[:, 1, 1:2]; m12 = M[:, 1, 2:3]
        m20 = M[:, 2, 0:1]; m21 = M[:, 2, 1:2]; m22 = M[:, 2, 2:3]
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
        """Central-difference sqrt|det J(flow)| at integer points c1. (K,) → (K,)."""
        x_p = (c1x + 1).clamp_max(W_a - 1)
        x_m = (c1x - 1).clamp_min(0)
        y_p = (c1y + 1).clamp_max(H_a - 1)
        y_m = (c1y - 1).clamp_min(0)
        af_flat = af.reshape(2, -1)

        def g(yy, xx):
            return af_flat[:, yy * W_a + xx]
        dx = g(c1y, x_p) - g(c1y, x_m)
        dy = g(y_p, c1x) - g(y_m, c1x)
        nx = (x_p - x_m).clamp_min(1).float()
        ny = (y_p - y_m).clamp_min(1).float()
        du_dx = dx[0] / nx; dv_dx = dx[1] / nx
        du_dy = dy[0] / ny; dv_dy = dy[1] / ny
        det = du_dx * dv_dy - du_dy * dv_dx
        return det.abs().clamp_min(1e-16).sqrt()

    def _window_sizes(self, sigma):
        """Pick (w1, w2) sizes from sigma. sigma is (..., K) float. Returns 4 tensors
        of the same shape — the desired window widths/heights in img_a and img_b.

        Replays the Phase A branch: small (0.2<σ<1) enlarges w1 by 1/σ; large
        (1≤σ<5) enlarges w2 by σ. Sigma is clamped here so out-of-range
        candidates don't blow up window sizes — the scale_ok mask filters them
        out before they're scored.
        """
        cW, cH = self.crop_W, self.crop_H
        sigma_c = sigma.clamp(min=0.21, max=4.99)
        small = sigma_c < 1
        inv = 1.0 / sigma_c
        sw1 = torch.where(small, cW * inv, torch.full_like(sigma_c, cW))
        sh1 = torch.where(small, cH * inv, torch.full_like(sigma_c, cH))
        sw2 = torch.where(small, torch.full_like(sigma_c, cW), cW * sigma_c)
        sh2 = torch.where(small, torch.full_like(sigma_c, cH), cH * sigma_c)
        return sw1, sh1, sw2, sh2

    @staticmethod
    def _window_vec(cx, cy, sw, sh, W, H):
        """Vectorized version of Phase A's `_window` clamping policy.

        cx, cy: (..., K) long.
        sw, sh: (..., K) float — desired widths/heights.
        W, H:   long tensors broadcastable to cx (typically (..., 1)).

        Returns four long tensors (..., K) — the half-open [xa, xb) × [ya, yb)
        crop windows per candidate, clamped to lie inside the image.
        """
        sw_i = (0.5 + sw).long()
        sh_i = (0.5 + sh).long()
        xa = cx - (0.5 + sw / 2).long()
        xb = xa + sw_i
        shift = (-xa).clamp_min(0)
        xa = xa + shift; xb = xb + shift
        shift = (xb - W).clamp_min(0)
        xa = xa - shift; xb = xb - shift
        bad = xa < 0
        xa = torch.where(bad, torch.zeros_like(xa), xa)
        xb = torch.where(bad, W.expand_as(xb), xb)

        ya = cy - (0.5 + sh / 2).long()
        yb = ya + sh_i
        shift = (-ya).clamp_min(0)
        ya = ya + shift; yb = yb + shift
        shift = (yb - H).clamp_min(0)
        ya = ya - shift; yb = yb - shift
        bad = ya < 0
        ya = torch.where(bad, torch.zeros_like(ya), ya)
        yb = torch.where(bad, H.expand_as(yb), yb)
        return xa, ya, xb, yb

    # ────────────────── scoring (vectorized over N × K) ───────────────────────

    def _score_analytic_vec(self, xa, ya, xb, yb, x2a, y2a, x2b, y2b, M_ab, device):
        """Score (N, K) analytic candidates: sample an M×M grid inside each w1,
        apply M_ab to those samples, score coverage in w2."""
        N, K = xa.shape
        M = self.SCORE_M
        us = (torch.arange(M, device=device, dtype=torch.float32) + 0.5) / M
        w1_w = (xb - xa).float()
        w1_h = (yb - ya).float()
        x_samp = xa[:, :, None, None].float() + us[None, None, None, :] * w1_w[:, :, None, None]
        y_samp = ya[:, :, None, None].float() + us[None, None, :, None] * w1_h[:, :, None, None]
        x_samp = x_samp.expand(N, K, M, M)
        y_samp = y_samp.expand(N, K, M, M)

        m00 = M_ab[:, 0, 0][:, None, None, None]; m01 = M_ab[:, 0, 1][:, None, None, None]; m02 = M_ab[:, 0, 2][:, None, None, None]
        m10 = M_ab[:, 1, 0][:, None, None, None]; m11 = M_ab[:, 1, 1][:, None, None, None]; m12 = M_ab[:, 1, 2][:, None, None, None]
        m20 = M_ab[:, 2, 0][:, None, None, None]; m21 = M_ab[:, 2, 1][:, None, None, None]; m22 = M_ab[:, 2, 2][:, None, None, None]
        wh = m20 * x_samp + m21 * y_samp + m22
        x2 = (m00 * x_samp + m01 * y_samp + m02) / wh
        y2 = (m10 * x_samp + m11 * y_samp + m12) / wh
        del wh, x_samp, y_samp
        return self._coverage_score(x2, y2, x2a, y2a, x2b, y2b, mask_in=None)

    def _score_flow_vec(self, xa, ya, xb, yb, x2a, y2a, x2b, y2b, af, mk, device):
        """Score (1, K) flow candidates by sampling af + mk at an M×M grid
        inside each w1 via a single grid_sample call."""
        N, K = xa.shape
        M = self.SCORE_M
        H_full, W_full = af.shape[-2], af.shape[-1]
        us = (torch.arange(M, device=device, dtype=torch.float32) + 0.5) / M
        w1_w = (xb - xa).float()
        w1_h = (yb - ya).float()
        x_samp = xa[:, :, None, None].float() + us[None, None, None, :] * w1_w[:, :, None, None]
        y_samp = ya[:, :, None, None].float() + us[None, None, :, None] * w1_h[:, :, None, None]
        x_samp = x_samp.expand(N, K, M, M)
        y_samp = y_samp.expand(N, K, M, M)

        norm_x = 2.0 * (x_samp + 0.5) / W_full - 1.0
        norm_y = 2.0 * (y_samp + 0.5) / H_full - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).reshape(1, N * K * M, M, 2)
        sampled = F.grid_sample(
            af.unsqueeze(0), grid,
            mode='nearest', padding_mode='zeros', align_corners=False,
        ).squeeze(0).reshape(2, N, K, M, M)
        x2 = sampled[0]; y2 = sampled[1]
        mk_sampled = F.grid_sample(
            mk.float().unsqueeze(0).unsqueeze(0), grid,
            mode='nearest', padding_mode='zeros', align_corners=False,
        ).reshape(N, K, M, M).bool()
        return self._coverage_score(x2, y2, x2a, y2a, x2b, y2b, mask_in=mk_sampled)

    @staticmethod
    def _coverage_score(x2, y2, x2a, y2a, x2b, y2b, mask_in):
        """Phase A's score: min(coverage_in_w2, occupancy_in_16x16_grid).

        x2, y2: (N, K, M, M) float — mapped sample positions in img_b.
        x2a, ...: (N, K) long — w2 bounds.
        mask_in: (N, K, M, M) bool or None — extra per-sample validity (flow).
        """
        N, K, M, _ = x2.shape
        device = x2.device
        x2a_b = x2a[:, :, None, None]; y2a_b = y2a[:, :, None, None]
        x2b_b = x2b[:, :, None, None]; y2b_b = y2b[:, :, None, None]

        x2i = x2.to(torch.long); y2i = y2.to(torch.long)
        in_w2 = ((x2i >= x2a_b) & (x2i < x2b_b) &
                 (y2i >= y2a_b) & (y2i < y2b_b))
        valid = in_w2 & mask_in if mask_in is not None else in_w2
        del in_w2
        score1 = valid.float().mean(dim=(2, 3))

        denom_x = (x2b_b - x2a_b).clamp_min(1)
        denom_y = (y2b_b - y2a_b).clamp_min(1)
        qx = (16 * (x2i - x2a_b) // denom_x).clamp_(0, 15)
        qy = (16 * (y2i - y2a_b) // denom_y).clamp_(0, 15)
        del x2i, y2i
        bucket = qy * 16 + qx
        del qx, qy
        # Sentinel 256 for invalid samples; one_hot[..., 256] is discarded.
        bucket = torch.where(valid, bucket, torch.full_like(bucket, 256))
        del valid
        bucket = bucket.reshape(N, K, -1)
        one_hot = torch.zeros(N, K, 257, device=device, dtype=torch.bool)
        one_hot.scatter_(2, bucket, True)
        del bucket
        score2 = one_hot[:, :, :256].float().mean(dim=2)

        return torch.minimum(score1, score2)

    # ────────────────── output construction ───────────────────────────────────

    def _build_outputs_analytic_batched(self, xa, ya, xb, yb, x2a, y2a, x2b, y2b,
                                          sa2ia, sb2ib, M_ab, good, device):
        """Build (N, ...) outputs for the analytic batch. xa…y2b are (N,) long
        (best window per item). good is (N,) bool — where False, emit identity
        M and NaN aflow / zero mask."""
        N = xa.shape[0]
        cW, cH = self.crop_W, self.crop_H

        Wwa = (xb - xa).float(); Hwa = (yb - ya).float()
        Wwb = (x2b - x2a).float(); Hwb = (y2b - y2a).float()
        Win_a = self._diag_translate_batch(Wwa / cW, Hwa / cH, xa.float(), ya.float(), device)
        Win_b = self._diag_translate_batch(Wwb / cW, Hwb / cH, x2a.float(), y2a.float(), device)

        M_a = torch.linalg.solve(sa2ia, Win_a)
        M_b = torch.linalg.solve(sb2ib, Win_b)

        j = torch.arange(cW, device=device, dtype=torch.float32)
        i = torch.arange(cH, device=device, dtype=torch.float32)
        gi, gj = torch.meshgrid(i, j, indexing='ij')
        x_a = xa[:, None, None].float() + gj[None] * (Wwa / cW)[:, None, None]
        y_a = ya[:, None, None].float() + gi[None] * (Hwa / cH)[:, None, None]

        m00 = M_ab[:, 0, 0][:, None, None]; m01 = M_ab[:, 0, 1][:, None, None]; m02 = M_ab[:, 0, 2][:, None, None]
        m10 = M_ab[:, 1, 0][:, None, None]; m11 = M_ab[:, 1, 1][:, None, None]; m12 = M_ab[:, 1, 2][:, None, None]
        m20 = M_ab[:, 2, 0][:, None, None]; m21 = M_ab[:, 2, 1][:, None, None]; m22 = M_ab[:, 2, 2][:, None, None]
        wh = m20 * x_a + m21 * y_a + m22
        x_b = (m00 * x_a + m01 * y_a + m02) / wh
        y_b = (m10 * x_a + m11 * y_a + m12) / wh
        x_b = (x_b - x2a[:, None, None].float()) * (cW / Wwb[:, None, None])
        y_b = (y_b - y2a[:, None, None].float()) * (cH / Hwb[:, None, None])
        aflow = torch.stack([x_b, y_b], dim=1)
        mask = torch.ones(N, cH, cW, device=device, dtype=torch.uint8)

        # Apply per-item validity mask without an early-return sync.
        eye = torch.eye(3, device=device, dtype=torch.float32)
        nan_aflow = torch.full_like(aflow, float('nan'))
        zero_mask = torch.zeros_like(mask)
        M_a = torch.where(good[:, None, None], M_a, eye)
        M_b = torch.where(good[:, None, None], M_b, eye)
        aflow = torch.where(good[:, None, None, None], aflow, nan_aflow)
        mask = torch.where(good[:, None, None], mask, zero_mask)
        return M_a, M_b, aflow, mask

    def _build_outputs_flow(self, xa, ya, xb, yb, x2a, y2a, x2b, y2b,
                             sa2ia, sb2ib, af, mk, device):
        """Build (3, 3) M_a, M_b and (2, cH, cW) aflow, (cH, cW) mask for one
        flow item. xa…y2b are 0-d long tensors."""
        cW, cH = self.crop_W, self.crop_H

        Wwa = (xb - xa).float(); Hwa = (yb - ya).float()
        Wwb = (x2b - x2a).float(); Hwb = (y2b - y2a).float()
        Win_a = self._diag_translate(Wwa / cW, Hwa / cH, xa.float(), ya.float(), device)
        Win_b = self._diag_translate(Wwb / cW, Hwb / cH, x2a.float(), y2a.float(), device)

        M_a = torch.linalg.solve(sa2ia, Win_a)
        M_b = torch.linalg.solve(sb2ib, Win_b)

        j = torch.arange(cW, device=device, dtype=torch.float32)
        i = torch.arange(cH, device=device, dtype=torch.float32)
        gi, gj = torch.meshgrid(i, j, indexing='ij')
        x_a = xa.float() + gj * (Wwa / cW)
        y_a = ya.float() + gi * (Hwa / cH)

        H_full, W_full = af.shape[-2], af.shape[-1]
        norm_x = 2.0 * (x_a + 0.5) / W_full - 1.0
        norm_y = 2.0 * (y_a + 0.5) / H_full - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            af.unsqueeze(0), grid,
            mode='nearest', padding_mode='zeros', align_corners=False,
        ).squeeze(0)
        x_b, y_b = sampled[0], sampled[1]
        mask_w = F.grid_sample(
            mk.float().unsqueeze(0).unsqueeze(0), grid,
            mode='nearest', padding_mode='zeros', align_corners=False,
        ).squeeze(0).squeeze(0).to(torch.uint8)

        x_b = (x_b - x2a.float()) * (cW / Wwb)
        y_b = (y_b - y2a.float()) * (cH / Hwb)
        invalid = ~mask_w.bool()
        x_b = torch.where(invalid, torch.full_like(x_b, float('nan')), x_b)
        y_b = torch.where(invalid, torch.full_like(y_b, float('nan')), y_b)
        aflow_crop = torch.stack([x_b, y_b], dim=0)
        return M_a, M_b, aflow_crop, mask_w

    @staticmethod
    def _diag_translate_batch(a, b, c, d, device):
        """[[a, 0, c], [0, b, d], [0, 0, 1]] for (N,) inputs → (N, 3, 3)."""
        N = a.shape[0]
        out = torch.zeros(N, 3, 3, device=device, dtype=torch.float32)
        out[:, 0, 0] = a; out[:, 1, 1] = b
        out[:, 0, 2] = c; out[:, 1, 2] = d
        out[:, 2, 2] = 1.0
        return out

    @staticmethod
    def _diag_translate(a, b, c, d, device):
        """[[a, 0, c], [0, b, d], [0, 0, 1]] for 0-d inputs → (3, 3)."""
        out = torch.zeros(3, 3, device=device, dtype=torch.float32)
        out[0, 0] = a; out[1, 1] = b
        out[0, 2] = c; out[1, 2] = d
        out[2, 2] = 1.0
        return out

    def _degenerate(self, device):
        cW, cH = self.crop_W, self.crop_H
        M_a = torch.eye(3, device=device, dtype=torch.float32)
        M_b = torch.eye(3, device=device, dtype=torch.float32)
        aflow = torch.full((2, cH, cW), float('nan'), device=device, dtype=torch.float32)
        mask = torch.zeros(cH, cW, device=device, dtype=torch.uint8)
        return M_a, M_b, aflow, mask
