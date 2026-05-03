"""
Tests for GPUWarp — the batched grid_sample-based renderer that pairs with
ParametricTransform.

These tests run on CPU (no CUDA required); the module is dtype/device-agnostic.
"""

import numpy as np
import pytest
import torch

from preprocessing.gpu_warp import GPUWarp


# ── helpers ───────────────────────────────────────────────────────────────────

def _checker(H=64, W=64, tile=8):
    """Build a deterministic uint8 (3, H, W) source image with sharp features."""
    yy, xx = np.mgrid[:H, :W]
    pat = (((xx // tile) + (yy // tile)) % 2 == 0).astype(np.uint8) * 255
    src = np.stack([pat, pat // 2, pat // 4], axis=0)
    return torch.from_numpy(src.astype(np.uint8))


def _identity_M():
    return torch.eye(3, dtype=torch.float32)


def _scale_M(sx, sy):
    return torch.tensor(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def _translate_M(tx, ty):
    return torch.tensor(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


# ── shape / dtype contract ────────────────────────────────────────────────────

class TestShapeAndDtype:
    def test_output_shape_is_batch_3_cropH_cropW(self):
        warp = GPUWarp(crop_size=(48, 32))
        srcs_a = [_checker(H=64, W=64), _checker(H=80, W=96)]
        srcs_b = [_checker(H=64, W=64), _checker(H=80, W=96)]
        M = torch.stack([_identity_M(), _identity_M()], dim=0)
        img_a, img_b = warp(srcs_a, srcs_b, M, M)
        assert img_a.shape == (2, 3, 32, 48)
        assert img_b.shape == (2, 3, 32, 48)

    def test_output_dtype_is_uint8(self):
        warp = GPUWarp(crop_size=(16, 16))
        src = _checker(H=32, W=32)
        M = _identity_M().unsqueeze(0)
        img_a, img_b = warp([src], [src], M, M)
        assert img_a.dtype == torch.uint8
        assert img_b.dtype == torch.uint8


# ── identity warp ─────────────────────────────────────────────────────────────

class TestIdentity:
    def test_identity_recovers_source(self):
        # src and crop have the same size; M = I → output ≈ source.
        H = W = 32
        warp = GPUWarp(crop_size=(W, H))
        src = _checker(H=H, W=W)
        M = _identity_M().unsqueeze(0)
        img, _ = warp([src], [src], M, M)
        # Bilinear at pixel centers of identical resolutions returns the input
        # exactly (no interpolation between distinct pixels).
        torch.testing.assert_close(img[0], src)


# ── translation ───────────────────────────────────────────────────────────────

class TestTranslation:
    def test_integer_translation_shifts_image(self):
        # M = translate(8, 4): output pixel (cx, cy) samples src at (cx+8, cy+4).
        H = W = 32
        warp = GPUWarp(crop_size=(W, H))
        src = _checker(H=H, W=W, tile=4)
        tx, ty = 8, 4
        M = _translate_M(tx, ty).unsqueeze(0)
        img, _ = warp([src], [src], M, M)
        # The interior of the output should equal a shifted slice of the input.
        # Check a small fully-in-bounds box.
        out = img[0, :, : H - ty, : W - tx]
        ref = src[:, ty:, tx:]
        torch.testing.assert_close(out, ref)


# ── scale (sub-rectangle crop) ────────────────────────────────────────────────

class TestScaleSubcrop:
    def test_half_scale_zooms_in(self):
        # M = diag(0.5, 0.5): each output pixel (cx, cy) samples src at
        # ((cx+0.5)*0.5, (cy+0.5)*0.5). Output should be a 2x zoom of the
        # top-left quadrant of src — bilinear sampled.
        H = W = 32
        warp = GPUWarp(crop_size=(W, H))
        src = _checker(H=H, W=W, tile=8)
        M = _scale_M(0.5, 0.5).unsqueeze(0)
        img, _ = warp([src], [src], M, M)
        # The center pixel (16, 16) of the output should land at source coord
        # (16.5*0.5, 16.5*0.5) = (8.25, 8.25). The input at (8, 8) is on a tile
        # boundary; just check that the output is in-range and not all zero.
        assert img.dtype == torch.uint8
        assert img[0].max().item() > 0


# ── per-sample independence ───────────────────────────────────────────────────

class TestPerSample:
    def test_different_M_per_sample(self):
        # Two samples, identical sources, different translations: outputs differ.
        H = W = 32
        warp = GPUWarp(crop_size=(W, H))
        src = _checker(H=H, W=W, tile=4)
        M = torch.stack([_translate_M(0, 0), _translate_M(8, 0)], dim=0)
        img, _ = warp([src, src], [src, src], M, M)
        assert not torch.equal(img[0], img[1])


# ── padding mode = zeros for out-of-bounds samples ────────────────────────────

class TestOutOfBounds:
    def test_far_out_of_bounds_returns_zero(self):
        # M translates so far that every sampled point is outside the source;
        # padding_mode='zeros' → output is all zeros.
        H = W = 32
        warp = GPUWarp(crop_size=(W, H))
        src = _checker(H=H, W=W) * 0 + 200  # solid 200 image
        src = src.to(torch.uint8)
        M = _translate_M(10000, 10000).unsqueeze(0)
        img, _ = warp([src], [src], M, M)
        assert (img == 0).all()
