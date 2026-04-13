"""
Tests for evaluation utilities: mnn_matcher and compute_metrics.

Both the HPatches format (6 images per sequence, homographies as GT) and the
flow format (2 images per pair, aflow as GT) are exercised.  The mock data
never touches real datasets or a real model — predictions are hand-crafted
tensors that make the expected outcome easy to reason about.

Design notes
------------
* Orthogonal unit descriptors guarantee that MNN returns the "obvious" match
  (each descriptor matches exactly itself) without depending on random seeds.
* Identity homography / identity aflow means reprojection error is zero, so
  MMA should be exactly 1.0 for a perfect-match scenario.
* The autouse fixture resets global accumulation state around every test so
  tests cannot bleed into each other.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction

from models.evaluation.compute_metrics import compute_metrics, mnn_matcher, _reset_state

# Number of descriptor dimensions used by the real model; keep consistent with
# the patchnet output so that the feature layout [x, y, scale, desc...] matches.
DESC_DIM = 128
FEAT_COLS = 3 + DESC_DIM  # x, y, scale, descriptors


# ── shared helpers ────────────────────────────────────────────────────────────

def orthogonal_features(n: int, x_start: float = 1.0, y_start: float = 1.0,
                        step: float = 5.0) -> torch.Tensor:
    """
    Build N feature rows with layout [x, y, scale=1, descriptor].
    The descriptor part uses the N standard-basis vectors, which are mutually
    orthogonal unit vectors.  MNN matching on these will always recover the
    identity permutation.
    """
    feats = torch.zeros(n, FEAT_COLS)
    feats[:, 0] = torch.arange(n) * step + x_start   # x
    feats[:, 1] = torch.arange(n) * step + y_start   # y
    feats[:, 2] = 1.0                                  # scale
    for i in range(min(n, DESC_DIM)):
        feats[i, 3 + i] = 1.0                         # e_i unit vector
    return feats


def identity_aflow(H: int, W: int) -> torch.Tensor:
    """Return (2, H, W) aflow where every pixel maps to itself."""
    xs = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    ys = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    return torch.stack([xs, ys], dim=0)


def hpatches_pred(n: int = 6, homography: torch.Tensor | None = None) -> EvalPrediction:
    """EvalPrediction in HPatches format: 6 images, identity homographies."""
    feats = orthogonal_features(n)
    if homography is None:
        homography = torch.eye(3)
    labels = torch.stack([torch.eye(3)] + [homography] * 5, dim=0)  # (6, 3, 3)
    return EvalPrediction(predictions=[feats] * 6, label_ids=labels)


def flow_pred(n: int = 6, H: int = 64, W: int = 64,
              aflow: torch.Tensor | None = None) -> EvalPrediction:
    """EvalPrediction in flow format: 2 images, identity aflow."""
    feats = orthogonal_features(n)
    if aflow is None:
        aflow = identity_aflow(H, W)
    return EvalPrediction(
        predictions=[feats, feats.clone()],
        label_ids=aflow,
    )


# ── autouse fixture ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    """Reset global accumulation state before and after every test."""
    _reset_state()
    yield
    _reset_state()


# ── mnn_matcher ───────────────────────────────────────────────────────────────

class TestMnnMatcher:
    def test_identity_descriptors_match_themselves(self):
        n = 8
        desc = F.normalize(torch.eye(n, DESC_DIM), p=2, dim=1)
        matches = mnn_matcher(desc, desc.clone())
        assert matches is not None
        assert matches.shape == (n, 2)
        assert (matches[:, 0] == matches[:, 1]).all()

    def test_empty_query_returns_none(self):
        assert mnn_matcher(torch.zeros(0, DESC_DIM), torch.randn(5, DESC_DIM)) is None

    def test_empty_database_returns_none(self):
        assert mnn_matcher(torch.randn(5, DESC_DIM), torch.zeros(0, DESC_DIM)) is None

    def test_permuted_descriptors_recovered(self):
        # Swap two descriptors in the database — matcher should still find them.
        n = 6
        desc_a = F.normalize(torch.eye(n, DESC_DIM), p=2, dim=1)
        desc_b = desc_a[[1, 0, 2, 3, 4, 5]]   # swap rows 0 and 1
        matches = mnn_matcher(desc_a, desc_b)
        assert matches is not None
        # Verify each matched pair has the same descriptor (by dot product ≈ 1)
        for ia, ib in matches:
            sim = (desc_a[ia] * desc_b[ib]).sum().item()
            assert sim == pytest.approx(1.0, abs=1e-5)


# ── compute_metrics — HPatches format ─────────────────────────────────────────

class TestComputeMetricsHPatches:
    def test_perfect_match_mma_is_one(self):
        result = compute_metrics(hpatches_pred(), compute_result=True)
        assert result["MMA"] == pytest.approx(1.0)

    def test_all_threshold_keys_present(self):
        result = compute_metrics(hpatches_pred(), compute_result=True)
        for thr in range(1, 16):
            assert f"error_{thr}" in result
        for key in ("MMA", "avg_matches", "avg_feats", "min_feats", "max_feats"):
            assert key in result

    def test_all_returned_values_are_python_floats(self):
        # The Trainer expects plain Python scalars, not tensors, in the metrics dict.
        result = compute_metrics(hpatches_pred(), compute_result=True)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"

    def test_accumulates_correctly_across_two_batches(self):
        # Simulate two sequences evaluated in sequence (the last batch triggers
        # compute_result=True as the Trainer does with batch_eval_metrics=True).
        pred = hpatches_pred()
        compute_metrics(pred, compute_result=False)          # batch 1
        result = compute_metrics(pred, compute_result=True)  # batch 2 (last)
        # Both batches are perfect → MMA should still be 1.0
        assert result["MMA"] == pytest.approx(1.0)

    def test_large_reprojection_error_gives_zero_tight_threshold(self):
        # Keypoints in img_a at x=0; with identity homography the projection
        # is also x=0.  Put img_b keypoints at x=100 so every match is wrong.
        n = 5
        feats_a = orthogonal_features(n, x_start=0.0, y_start=0.0)
        feats_b = orthogonal_features(n, x_start=100.0, y_start=0.0)
        labels = torch.eye(3).unsqueeze(0).expand(6, 3, 3).clone()
        pred = EvalPrediction(
            predictions=[feats_a] + [feats_b] * 5,
            label_ids=labels,
        )
        result = compute_metrics(pred, compute_result=True)
        # dist ≈ 100 px → no match passes the 1-px threshold
        assert result["error_1"] == pytest.approx(0.0, abs=1e-6)

    def test_non_result_call_returns_none(self):
        assert compute_metrics(hpatches_pred(), compute_result=False) is None


# ── compute_metrics — flow format ─────────────────────────────────────────────

class TestComputeMetricsFlow:
    def test_perfect_match_identity_flow_mma_is_one(self):
        result = compute_metrics(flow_pred(), compute_result=True)
        assert result["MMA"] == pytest.approx(1.0)

    def test_error_at_1px_is_one_for_zero_error(self):
        result = compute_metrics(flow_pred(), compute_result=True)
        assert result["error_1"] == pytest.approx(1.0)

    def test_all_threshold_keys_present(self):
        result = compute_metrics(flow_pred(), compute_result=True)
        for thr in range(1, 16):
            assert f"error_{thr}" in result
        assert "MMA" in result

    def test_all_returned_values_are_python_floats(self):
        result = compute_metrics(flow_pred(), compute_result=True)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"

    def test_nan_aflow_regions_are_skipped_without_error(self):
        # All-NaN aflow → no valid GT correspondences → errors stay 0, no crash.
        H, W = 32, 32
        aflow = torch.full((2, H, W), float("nan"))
        result = compute_metrics(flow_pred(aflow=aflow), compute_result=True)
        assert all(isinstance(v, float) and np.isfinite(v) for v in result.values())
        assert result["MMA"] == pytest.approx(0.0)

    def test_numpy_aflow_accepted(self):
        # compute_metrics should handle both torch.Tensor and np.ndarray labels.
        H, W = 32, 32
        aflow_np = identity_aflow(H, W).numpy()
        pred = EvalPrediction(
            predictions=[orthogonal_features(4)] * 2,
            label_ids=aflow_np,
        )
        result = compute_metrics(pred, compute_result=True)
        assert result["MMA"] == pytest.approx(1.0)

    def test_no_keypoints_gives_zero_mma(self):
        empty = torch.zeros(0, FEAT_COLS)
        pred = EvalPrediction(
            predictions=[empty, empty.clone()],
            label_ids=identity_aflow(32, 32),
        )
        result = compute_metrics(pred, compute_result=True)
        assert result["MMA"] == pytest.approx(0.0)


# ── state reset ───────────────────────────────────────────────────────────────

class TestStateReset:
    def test_two_sequential_evals_are_independent(self):
        # Eval 1: perfect match → MMA = 1.0, then state resets automatically.
        result1 = compute_metrics(flow_pred(), compute_result=True)

        # Eval 2: no keypoints → MMA = 0.0.  If state wasn't reset, accumulated
        # hits from eval 1 would bleed in and raise MMA above 0.
        empty = torch.zeros(0, FEAT_COLS)
        pred2 = EvalPrediction(
            predictions=[empty, empty.clone()],
            label_ids=identity_aflow(32, 32),
        )
        result2 = compute_metrics(pred2, compute_result=True)

        assert result1["MMA"] == pytest.approx(1.0)
        assert result2["MMA"] == pytest.approx(0.0)

    def test_explicit_reset_clears_partial_accumulation(self):
        compute_metrics(hpatches_pred(), compute_result=False)  # partial accumulation
        _reset_state()                                           # discard it
        # Now a fresh single-batch eval should reflect only its own data.
        result = compute_metrics(hpatches_pred(), compute_result=True)
        assert result["MMA"] == pytest.approx(1.0)
