"""
Tests for ParametricTransform — the CPU-side preprocessor that samples
geometric transform parameters and crop windows without rendering, leaving
the actual image warping to GPUWarp on the GPU.

All tests use synthetic PIL images so no real dataset files are required.

Flow image encoding
-------------------
Optical flow is stored as an RGBA uint8 PIL image where the 4 bytes per pixel
are reinterpreted as 2 × int16 (x-flow, y-flow).  A zero-filled RGBA image
therefore encodes zero flow, meaning aflow[y, x] == (x, y) (each pixel maps
to itself).
"""

import pytest
import numpy as np
import torch
from PIL import Image

from preprocessing.transform_builder import ParametricTransform


# ── image factories ───────────────────────────────────────────────────────────

def rgb_image(w: int = 400, h: int = 400) -> Image.Image:
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def flow_image(h: int = 400, w: int = 400) -> Image.Image:
    """RGBA uint8 whose bytes decode to (H, W, 2) int16 = zero flow."""
    return Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), "RGBA")


def mask_image(h: int = 400, w: int = 400, fill: int = 255) -> Image.Image:
    return Image.fromarray(np.full((h, w), fill, dtype=np.uint8), "L")


# ── ParametricTransform ───────────────────────────────────────────────────────

class TestParametricTransform:
    """
    ParametricTransform is the CPU-side preprocessor that defers image
    rendering to the GPU. It returns raw uint8 source images, two 3×3
    M_a/M_b homographies (crop pixel → source pixel), the cropped aflow
    in float32, and a uint8 mask.
    """

    @pytest.fixture
    def transform(self):
        return ParametricTransform(crop_size=(192, 192))

    @staticmethod
    def _batch(image=None, im0=None, im1=None, flow=None, mask=None):
        return {
            "image":    [image],
            "im0.jpg":  [im0],
            "im1.jpg":  [im1],
            "flow.png": [flow],
            "mask.png": [mask],
        }

    def test_synthetic_mode_emits_required_keys(self, transform):
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        assert {"src_a", "src_b", "M_a", "M_b", "aflow", "mask"} <= out.keys()

    def test_still_mode_emits_required_keys(self, transform):
        batch = self._batch(im0=rgb_image(400, 400), im1=rgb_image(400, 400))
        out = transform(batch)
        assert {"src_a", "src_b", "M_a", "M_b", "aflow", "mask"} <= out.keys()

    def test_flow_mode_emits_required_keys(self, transform):
        H = W = 400
        batch = self._batch(im0=rgb_image(W, H), im1=rgb_image(W, H),
                            flow=flow_image(H, W), mask=mask_image(H, W))
        out = transform(batch)
        assert {"src_a", "src_b", "M_a", "M_b", "aflow", "mask"} <= out.keys()

    def test_src_a_is_a_uint8_rgb_tensor(self, transform):
        # src_a is the raw (un-rendered) source — variable resolution.
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        src_a = out["src_a"][0]
        assert isinstance(src_a, torch.Tensor)
        assert src_a.dtype == torch.uint8
        assert src_a.ndim == 3
        assert src_a.shape[0] == 3

    def test_M_matrices_are_3x3_float32(self, transform):
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        M_a, M_b = out["M_a"][0], out["M_b"][0]
        assert M_a.shape == (3, 3) and M_b.shape == (3, 3)
        assert M_a.dtype == torch.float32 and M_b.dtype == torch.float32

    def test_aflow_and_mask_match_crop_size(self, transform):
        # aflow is (2, crop_H, crop_W); mask is (crop_H, crop_W).
        crop_W, crop_H = (192, 192)
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        assert out["aflow"][0].shape == (2, crop_H, crop_W)
        assert out["mask"][0].shape == (crop_H, crop_W)

    def test_synthetic_M_a_corner_maps_to_source_corner(self, transform):
        # M_a maps crop pixel-corner (0,0) to a source pixel-corner inside
        # [0, W_src] × [0, H_src]. This sanity-checks that the chained
        # Win_a ∘ scale_a^-1 lives entirely inside the source image.
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        M_a = out["M_a"][0].numpy()
        src_a = out["src_a"][0]
        H_src, W_src = src_a.shape[1], src_a.shape[2]
        # Apply M_a to the four crop corners
        corners = np.array([
            [0, 0, 1],
            [192, 0, 1],
            [0, 192, 1],
            [192, 192, 1],
        ], dtype=np.float64).T
        src_corners = M_a @ corners
        src_corners = src_corners[:2] / src_corners[2:]
        # All four corners should map inside the source image with a small slack
        assert (src_corners[0] >= -1e-3).all() and (src_corners[0] <= W_src + 1e-3).all()
        assert (src_corners[1] >= -1e-3).all() and (src_corners[1] <= H_src + 1e-3).all()

    def test_still_mode_M_a_is_identity_window(self, transform):
        # In still mode, src_a == im0 with no transform; M_a is just Win_a.
        # That means the rotational/perspective parts of M_a must be zero
        # (only a 2D scale + translation).
        batch = self._batch(im0=rgb_image(400, 400), im1=rgb_image(400, 400))
        out = transform(batch)
        M_a = out["M_a"][0].numpy()
        # Off-diagonal upper 2x2 entries and the bottom row must be zero.
        assert M_a[0, 1] == pytest.approx(0.0)
        assert M_a[1, 0] == pytest.approx(0.0)
        assert M_a[2, 0] == pytest.approx(0.0)
        assert M_a[2, 1] == pytest.approx(0.0)
        assert M_a[2, 2] == pytest.approx(1.0)
