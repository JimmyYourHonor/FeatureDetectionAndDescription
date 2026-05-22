"""
Tests for GPUWindowSelect — the GPU module that replaces the CPU trial-loop
crop selection. Consumes the lightweight per-sample chain emitted by
ParametricTransform and produces the per-sample M_a / M_b / aflow / mask
tensors that GPUWarp + the loss expect.

These tests run on CPU (the module is plain torch ops; no CUDA required).
"""

import numpy as np
import pytest
import torch

from preprocessing.gpu_window_select import GPUWindowSelect, MODE_ANALYTIC, MODE_FLOW


CROP = 64  # smaller than the production crop to keep tests cheap


@pytest.fixture
def select():
    return GPUWindowSelect(crop_size=(CROP, CROP), n_samples=2, max_attempts_per_sample=10)


# ── output contracts ──────────────────────────────────────────────────────────

def _analytic_batch(B=2, W=200, H=200):
    """A minimal analytic-mode batch: M_ab = identity, sources at (W, H)."""
    sa2ia = torch.eye(3).repeat(B, 1, 1)
    sb2ib = torch.eye(3).repeat(B, 1, 1)
    M_ab = torch.eye(3).repeat(B, 1, 1)
    img_size = torch.tensor([[W, H, W, H]] * B, dtype=torch.int32)
    mode = [MODE_ANALYTIC] * B
    aflow_full = [torch.zeros(2, 1, 1) for _ in range(B)]
    mask_full = [torch.ones(1, 1, dtype=torch.uint8) for _ in range(B)]
    return sa2ia, sb2ib, M_ab, img_size, mode, aflow_full, mask_full


def _flow_batch(B=1, W=200, H=200):
    """A minimal flow-mode batch with identity flow (img_a coord == img_b coord)."""
    sa2ia = torch.eye(3).repeat(B, 1, 1)
    sb2ib = torch.eye(3).repeat(B, 1, 1)
    M_ab = torch.eye(3).repeat(B, 1, 1)
    img_size = torch.tensor([[W, H, W, H]] * B, dtype=torch.int32)
    mode = [MODE_FLOW] * B
    # identity flow: aflow_full[:, y, x] == (x, y)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing='ij')
    af = torch.stack([xs, ys], dim=0)
    aflow_full = [af.clone() for _ in range(B)]
    mask_full = [torch.ones(H, W, dtype=torch.uint8) for _ in range(B)]
    return sa2ia, sb2ib, M_ab, img_size, mode, aflow_full, mask_full


class TestOutputShapesAndDtypes:
    def test_analytic_mode_returns_expected_shapes(self, select):
        torch.manual_seed(0)
        M_a, M_b, aflow, mask = select(*_analytic_batch(B=3))
        assert M_a.shape == (3, 3, 3) and M_a.dtype == torch.float32
        assert M_b.shape == (3, 3, 3) and M_b.dtype == torch.float32
        assert aflow.shape == (3, 2, CROP, CROP) and aflow.dtype == torch.float32
        assert mask.shape == (3, CROP, CROP) and mask.dtype == torch.uint8

    def test_flow_mode_returns_expected_shapes(self, select):
        torch.manual_seed(0)
        M_a, M_b, aflow, mask = select(*_flow_batch(B=2))
        assert M_a.shape == (2, 3, 3)
        assert aflow.shape == (2, 2, CROP, CROP)
        assert mask.shape == (2, CROP, CROP)


class TestDeterminism:
    def test_same_seed_produces_same_outputs(self, select):
        torch.manual_seed(123)
        out_a = select(*_analytic_batch(B=2))
        torch.manual_seed(123)
        out_b = select(*_analytic_batch(B=2))
        for a, b in zip(out_a, out_b):
            torch.testing.assert_close(a, b)


class TestAnalyticIdentity:
    """When M_ab is identity, img_a == img_b so a chosen window in img_a
    should map to itself in img_b: aflow_crop(j, i) ≈ (j, i)."""

    def test_aflow_crop_recovers_identity(self, select):
        torch.manual_seed(0)
        _, _, aflow, mask = select(*_analytic_batch(B=1, W=400, H=400))
        valid = mask[0].bool()
        assert valid.any(), "expected some valid pixels in the chosen window"
        x = aflow[0, 0][valid]
        y = aflow[0, 1][valid]
        # Build the expected (j, i) grid restricted to valid pixels.
        j = torch.arange(CROP).float().unsqueeze(0).expand(CROP, CROP)
        i = torch.arange(CROP).float().unsqueeze(1).expand(CROP, CROP)
        expect_x = j[valid]
        expect_y = i[valid]
        # Allow small discretization error from the window-quantization path.
        torch.testing.assert_close(x, expect_x, atol=1.0, rtol=0)
        torch.testing.assert_close(y, expect_y, atol=1.0, rtol=0)


class TestEmptyMaskFallback:
    def test_all_zero_mask_returns_degenerate_sample(self, select):
        torch.manual_seed(0)
        sa2ia, sb2ib, M_ab, img_size, mode, aflow_full, _ = _flow_batch(B=1, W=128, H=128)
        mask_full = [torch.zeros(128, 128, dtype=torch.uint8)]
        M_a, M_b, aflow, mask = select(sa2ia, sb2ib, M_ab, img_size, mode, aflow_full, mask_full)
        assert torch.equal(M_a[0], torch.eye(3))
        assert torch.isnan(aflow).all()
        assert (mask == 0).all()


# ── Phase B specifics ─────────────────────────────────────────────────────────

class TestMixedModeBatch:
    """Phase B batches analytic items across the batch axis and loops flow
    items per-item. Mixed batches must keep per-item outputs aligned."""

    def test_mixed_batch_preserves_per_item_outputs(self, select):
        """A two-item batch — one analytic with identity M_ab, one flow with
        zero mask — should produce non-degenerate output for the analytic
        item and degenerate output for the flow item, in the right slots."""
        torch.manual_seed(0)
        # Item 0: analytic, identity M_ab — should succeed.
        # Item 1: flow, zero mask — should fall back to degenerate.
        sa2ia = torch.eye(3).repeat(2, 1, 1)
        sb2ib = torch.eye(3).repeat(2, 1, 1)
        M_ab = torch.eye(3).repeat(2, 1, 1)
        img_size = torch.tensor([[400, 400, 400, 400], [128, 128, 128, 128]],
                                 dtype=torch.int32)
        mode = [MODE_ANALYTIC, MODE_FLOW]
        aflow_full = [torch.zeros(2, 1, 1), torch.zeros(2, 128, 128)]
        mask_full = [torch.ones(1, 1, dtype=torch.uint8),
                     torch.zeros(128, 128, dtype=torch.uint8)]

        M_a, M_b, aflow, mask = select(sa2ia, sb2ib, M_ab, img_size,
                                        mode, aflow_full, mask_full)
        # Item 0 (analytic): non-degenerate.
        assert mask[0].any()
        assert not torch.isnan(aflow[0]).all()
        # Item 1 (flow, empty mask): degenerate.
        assert torch.equal(M_a[1], torch.eye(3))
        assert torch.isnan(aflow[1]).all()
        assert (mask[1] == 0).all()


class TestBatchIndependence:
    """The vectorized analytic path must not bleed state across items in a
    batch — perturbing other items' M_ab/img_size must not change this
    item's output (a single shared torch.rand consumes the same RNG stream
    in both runs, so c1x/c1y are identical; scoring uses only this item's
    M_ab via the per-item gather)."""

    def test_other_items_dont_affect_this_item(self, select):
        sa2ia = torch.eye(3).repeat(3, 1, 1)
        sb2ib = torch.eye(3).repeat(3, 1, 1)
        img_size = torch.tensor([[400, 400, 400, 400]] * 3, dtype=torch.int32)
        mode = [MODE_ANALYTIC] * 3
        aflow_full = [torch.zeros(2, 1, 1) for _ in range(3)]
        mask_full = [torch.ones(1, 1, dtype=torch.uint8) for _ in range(3)]

        M_ab = torch.eye(3).repeat(3, 1, 1)
        torch.manual_seed(7)
        out_a = select(sa2ia, sb2ib, M_ab, img_size, mode, aflow_full, mask_full)

        # Perturb items 1 and 2 only. Item 0's M_ab is unchanged.
        M_ab_alt = M_ab.clone()
        M_ab_alt[1] = torch.tensor([[0.7, 0.0, 10.0],
                                     [0.0, 0.7, -5.0],
                                     [0.0, 0.0, 1.0]])
        M_ab_alt[2] = torch.tensor([[1.0, 0.3, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]])
        torch.manual_seed(7)
        out_b = select(sa2ia, sb2ib, M_ab_alt, img_size, mode, aflow_full, mask_full)

        for tensor_a, tensor_b in zip(out_a, out_b):
            torch.testing.assert_close(tensor_a[0], tensor_b[0])
