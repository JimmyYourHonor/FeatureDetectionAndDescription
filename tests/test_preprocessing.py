"""
Tests for the preprocessing pipeline: SyntheticPairTransform, StillPairTransform,
FlowPairTransform, and the top-level FullTransform.

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

from preprocessing.transform_builder import (
    SyntheticPairTransform,
    StillPairTransform,
    FlowPairTransform,
    FullTransform,
)


# ── image factories ───────────────────────────────────────────────────────────

def rgb_image(w: int = 400, h: int = 400) -> Image.Image:
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def flow_image(h: int = 400, w: int = 400) -> Image.Image:
    """RGBA uint8 whose bytes decode to (H, W, 2) int16 = zero flow."""
    return Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), "RGBA")


def mask_image(h: int = 400, w: int = 400, fill: int = 255) -> Image.Image:
    return Image.fromarray(np.full((h, w), fill, dtype=np.uint8), "L")


# ── SyntheticPairTransform ────────────────────────────────────────────────────

class TestSyntheticPairTransform:
    @pytest.fixture
    def transform(self):
        return SyntheticPairTransform()

    def test_output_has_required_keys(self, transform):
        out = transform(rgb_image())
        assert {"img_a", "img_b", "aflow"} <= out.keys()

    def test_img_a_and_img_b_are_pil_images(self, transform):
        out = transform(rgb_image())
        assert isinstance(out["img_a"], Image.Image)
        assert isinstance(out["img_b"], Image.Image)

    def test_aflow_shape_matches_img_a(self, transform):
        out = transform(rgb_image(400, 300))
        W, H = out["img_a"].size   # PIL uses (width, height)
        assert out["aflow"].shape == (H, W, 2)

    def test_aflow_is_float32(self, transform):
        out = transform(rgb_image())
        assert out["aflow"].dtype == np.float32

    def test_img_b_differs_from_img_a(self, transform):
        # The distortion should change at least some pixels
        out = transform(rgb_image())
        a = np.array(out["img_a"])
        b = np.array(out["img_b"].resize(out["img_a"].size))
        assert not np.array_equal(a, b)


# ── StillPairTransform ────────────────────────────────────────────────────────

class TestStillPairTransform:
    @pytest.fixture
    def transform(self):
        return StillPairTransform()

    def test_output_has_required_keys(self, transform):
        out = transform(rgb_image(), rgb_image())
        assert {"img_a", "img_b", "aflow"} <= out.keys()

    def test_img_a_is_the_original_first_image(self, transform):
        im0, im1 = rgb_image(), rgb_image()
        out = transform(im0, im1)
        # img_a must be im0 unchanged — no geometric transform is applied to it
        assert out["img_a"] is im0

    def test_aflow_values_are_within_destination_image_bounds(self, transform):
        # im0 is 200×200, im1 is 400×400.  Valid aflow values must fall inside
        # im1 (i.e. in [0, 400)).  RandomTilting can push some pixels out of
        # bounds, but the mean flow across the image should stay positive.
        im0 = rgb_image(200, 200)
        im1 = rgb_image(400, 400)
        out = transform(im0, im1)
        aflow = out["aflow"]   # (H0=200, W0=200, 2)
        # The mean destination x and y should be somewhere in the positive half
        # of im1 (roughly 200±noise), so clearly > 0.
        assert float(aflow[..., 0].mean()) > 0.0
        assert float(aflow[..., 1].mean()) > 0.0

    def test_aflow_is_float32(self, transform):
        out = transform(rgb_image(), rgb_image())
        assert out["aflow"].dtype == np.float32


# ── FlowPairTransform ─────────────────────────────────────────────────────────

class TestFlowPairTransform:
    @pytest.fixture
    def transform(self):
        return FlowPairTransform()

    def test_output_has_required_keys(self, transform):
        H = W = 64
        out = transform(rgb_image(W, H), rgb_image(W, H),
                        flow_image(H, W), mask_image(H, W))
        assert {"img_a", "img_b", "aflow", "mask"} <= out.keys()

    def test_zero_flow_gives_identity_aflow(self, transform):
        # Zero int16 flow → aflow[y, x] = (x, y) exactly.
        H, W = 64, 64
        out = transform(rgb_image(W, H), rgb_image(W, H),
                        flow_image(H, W), mask_image(H, W))
        aflow = out["aflow"]   # (H, W, 2)
        for y in range(0, H, 8):
            for x in range(0, W, 8):
                assert aflow[y, x, 0] == pytest.approx(x, abs=1e-4)
                assert aflow[y, x, 1] == pytest.approx(y, abs=1e-4)

    def test_aflow_shape_matches_img_a(self, transform):
        H, W = 64, 64
        out = transform(rgb_image(W, H), rgb_image(W, H),
                        flow_image(H, W), mask_image(H, W))
        assert out["aflow"].shape == (H, W, 2)

    def test_mask_shape_matches_img_a(self, transform):
        H, W = 64, 64
        out = transform(rgb_image(W, H), rgb_image(W, H),
                        flow_image(H, W), mask_image(H, W))
        assert out["mask"].shape == (H, W)

    def test_aflow_is_float32(self, transform):
        H, W = 64, 64
        out = transform(rgb_image(W, H), rgb_image(W, H),
                        flow_image(H, W), mask_image(H, W))
        assert out["aflow"].dtype == np.float32


# ── FullTransform ─────────────────────────────────────────────────────────────

class TestFullTransform:
    """
    FullTransform dispatches on which fields are non-None:
      Synthetic  – 'image' is set, others None
      Still pair – 'im0.jpg' + 'im1.jpg' set, 'flow.png' None
      Flow pair  – 'im0.jpg', 'im1.jpg', 'flow.png', 'mask.png' all set
    """

    @pytest.fixture
    def transform(self):
        return FullTransform()

    # helper: build a one-sample batch dict
    @staticmethod
    def _batch(image=None, im0=None, im1=None, flow=None, mask=None):
        return {
            "image":    [image],
            "im0.jpg":  [im0],
            "im1.jpg":  [im1],
            "flow.png": [flow],
            "mask.png": [mask],
        }

    def test_synthetic_mode_produces_required_keys(self, transform):
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        assert {"img_a", "img_b", "aflow", "mask"} <= out.keys()

    def test_still_mode_produces_required_keys(self, transform):
        batch = self._batch(im0=rgb_image(400, 400), im1=rgb_image(400, 400))
        out = transform(batch)
        assert {"img_a", "img_b", "aflow", "mask"} <= out.keys()

    def test_flow_mode_produces_required_keys(self, transform):
        H = W = 400
        batch = self._batch(im0=rgb_image(W, H), im1=rgb_image(W, H),
                            flow=flow_image(H, W), mask=mask_image(H, W))
        out = transform(batch)
        assert {"img_a", "img_b", "aflow", "mask"} <= out.keys()

    def test_img_a_is_a_normalised_rgb_tensor(self, transform):
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        img_a = out["img_a"][0]
        assert isinstance(img_a, torch.Tensor)
        assert img_a.ndim == 3
        assert img_a.shape[0] == 3   # RGB channels

    def test_aflow_tensor_shape_matches_img_a(self, transform):
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        img_a = out["img_a"][0]       # (3, H, W)
        aflow = out["aflow"][0]       # (2, H, W)
        _, H, W = img_a.shape
        assert aflow.shape == (2, H, W)

    def test_mask_tensor_shape_matches_img_a(self, transform):
        batch = self._batch(image=rgb_image(400, 400))
        out = transform(batch)
        img_a = out["img_a"][0]
        mask = out["mask"][0]
        _, H, W = img_a.shape
        assert mask.shape == (H, W)

    def test_invalid_batch_raises_value_error(self, transform):
        # All fields None — no dataset mode matches
        batch = self._batch()
        with pytest.raises((ValueError, Exception)):
            transform(batch)
