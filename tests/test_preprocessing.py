"""
Tests for ParametricTransform — the CPU-side preprocessor that samples
geometric transform parameters and emits a lightweight per-sample geometric
chain. Window selection, image rendering, and augmentation all run on GPU
downstream (see GPUWindowSelect, GPUWarp, GPUBatchAugment).

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
from preprocessing.gpu_window_select import MODE_ANALYTIC, MODE_FLOW


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

REQUIRED_KEYS = {
    "src_a", "src_b",
    "sa2ia", "sb2ib", "M_ab",
    "img_size", "mode",
    "aflow_full", "mask_full",
}


class TestParametricTransform:
    """
    ParametricTransform now emits raw uint8 sources, src→img and img_a→img_b
    homographies, an img_size record, a mode flag, and (in flow mode) the
    per-pixel flow + mask. It no longer materializes per-image aflow grids
    or runs the trial-loop crop selection — that's GPUWindowSelect's job.
    """

    @pytest.fixture
    def transform(self):
        return ParametricTransform()

    @staticmethod
    def _batch(image=None, im0=None, im1=None, flow=None, mask=None):
        return {
            "image":    [image],
            "im0.jpg":  [im0],
            "im1.jpg":  [im1],
            "flow.png": [flow],
            "mask.png": [mask],
        }

    def test_synthetic_mode_emits_required_keys_and_mode(self, transform):
        out = transform(self._batch(image=rgb_image(400, 400)))
        assert REQUIRED_KEYS <= out.keys()
        assert out["mode"][0] == MODE_ANALYTIC

    def test_still_mode_emits_required_keys_and_mode(self, transform):
        out = transform(self._batch(im0=rgb_image(400, 400), im1=rgb_image(400, 400)))
        assert REQUIRED_KEYS <= out.keys()
        assert out["mode"][0] == MODE_ANALYTIC

    def test_flow_mode_emits_required_keys_and_mode(self, transform):
        H = W = 400
        out = transform(self._batch(im0=rgb_image(W, H), im1=rgb_image(W, H),
                                    flow=flow_image(H, W), mask=mask_image(H, W)))
        assert REQUIRED_KEYS <= out.keys()
        assert out["mode"][0] == MODE_FLOW

    def test_src_a_is_a_uint8_rgb_tensor(self, transform):
        # src_a is the raw (un-rendered) source — variable resolution.
        out = transform(self._batch(image=rgb_image(400, 400)))
        src_a = out["src_a"][0]
        assert isinstance(src_a, torch.Tensor)
        assert src_a.dtype == torch.uint8
        assert src_a.ndim == 3
        assert src_a.shape[0] == 3

    def test_chain_matrices_are_3x3_float32(self, transform):
        out = transform(self._batch(image=rgb_image(400, 400)))
        for key in ("sa2ia", "sb2ib", "M_ab"):
            m = out[key][0]
            assert m.shape == (3, 3)
            assert m.dtype == torch.float32

    def test_img_size_records_img_a_and_img_b_dims(self, transform):
        out = transform(self._batch(im0=rgb_image(400, 400), im1=rgb_image(400, 400)))
        sz = out["img_size"][0]
        assert sz.shape == (4,)
        # img_a == im0, so first two entries equal im0 size
        assert sz[0].item() == 400 and sz[1].item() == 400

    def test_flow_mode_carries_real_flow_and_mask_at_img_a_resolution(self, transform):
        H, W = 320, 480
        out = transform(self._batch(
            im0=rgb_image(W, H), im1=rgb_image(W, H),
            flow=flow_image(H, W), mask=mask_image(H, W),
        ))
        af = out["aflow_full"][0]
        mk = out["mask_full"][0]
        assert af.shape == (2, H, W) and af.dtype == torch.float32
        assert mk.shape == (H, W) and mk.dtype == torch.uint8

    def test_analytic_mode_emits_aflow_placeholder(self, transform):
        # In analytic modes M_ab carries the mapping; aflow_full / mask_full
        # are tiny placeholders so the collator can still treat them uniformly.
        out = transform(self._batch(image=rgb_image(400, 400)))
        af = out["aflow_full"][0]
        mk = out["mask_full"][0]
        assert af.shape == (2, 1, 1)
        assert mk.shape == (1, 1)

    def test_still_mode_sa2ia_is_identity(self, transform):
        # Still mode keeps img_a == im0 unchanged, so src_a→img_a is identity.
        out = transform(self._batch(im0=rgb_image(400, 400), im1=rgb_image(400, 400)))
        sa2ia = out["sa2ia"][0].numpy()
        np.testing.assert_allclose(sa2ia, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_flow_mode_M_ab_is_identity_placeholder(self, transform):
        # Flow mode carries the real img_a→img_b mapping in aflow_full;
        # M_ab is left as identity so analytic-mode helpers don't trip.
        H = W = 400
        out = transform(self._batch(
            im0=rgb_image(W, H), im1=rgb_image(W, H),
            flow=flow_image(H, W), mask=mask_image(H, W),
        ))
        M_ab = out["M_ab"][0].numpy()
        np.testing.assert_allclose(M_ab, np.eye(3, dtype=np.float32), atol=1e-6)
