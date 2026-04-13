"""
Unit tests for model components.

Ideas for model testing — and why each matters
-----------------------------------------------
1.  Output shape contract
    The networks are fully-convolutional with dilated strides (no pooling that
    reduces resolution).  For any input size the descriptor / reliability /
    repeatability maps must have the same spatial dimensions as the input.
    Catching a regression here is cheap and immediate.

2.  Descriptor L2 normalisation
    BaseNet.normalize() calls F.normalize(x, p=2, dim=1), so every pixel in
    every output descriptor map must be a unit vector.  Testing this is a
    direct sanity-check of the output contract the loss functions rely on.

3.  Confidence map range [0, 1]
    Reliability uses a 2-channel softmax → [0,1]; repeatability uses softplus
    normalised to (0,1).  Both losses would silently break if these bounds
    were violated.

4.  Loss is finite
    Given a plausible forward pass the total MultiLoss must produce a finite
    scalar.  If any component produces NaN the training loop diverges silently.

5.  Gradient flow through all parameters
    After one backward pass every learnable parameter must have a non-None
    gradient.  A detach() or in-place op in the wrong place will break this.

6.  PeakyLoss extremes
    A uniform repeatability map should give loss ≈ 1.0 (max == avg, so no
    peakiness) and a single-spike map should give lower loss.  This verifies
    the loss pushes saliency in the right direction.

7.  Sampler gt labelling
    With maxpool_pos=True and an identity aflow the first column of gt must be
    all-ones (positive match) and all other columns must be zero (negatives).
    Incorrect labelling would silently corrupt the AP loss.

8.  NMS selects local maxima only
    A RepeatabilityMap with a single peak must produce exactly one keypoint at
    the correct location.  Pixels below the threshold must produce nothing.
"""

import pytest
import torch
import torch.nn.functional as F

from models.nets.patchnet import Quad_L2Net_ConfCFS
from models.nets.convnextv2 import ConvNeXtV2
from models.loss.losses import MultiLoss
from models.loss.reliability_loss import ReliabilityLoss
from models.loss.repeatability_loss import CosimLoss, PeakyLoss
from models.sampler.sampler import NghSampler2
from models.evaluation.utils import NonMaxSuppression


# ── shared helpers ────────────────────────────────────────────────────────────

def identity_aflow(B: int, H: int, W: int) -> torch.Tensor:
    """(B, 2, H, W) aflow where every pixel maps to itself."""
    xs = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    ys = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    return torch.stack([xs, ys], dim=0).unsqueeze(0).expand(B, 2, H, W).clone()


def random_batch(B: int = 2, D: int = 128, H: int = 48, W: int = 48):
    """Produce a dict of model outputs + training targets for loss tests."""
    descriptors = [F.normalize(torch.randn(B, D, H, W), p=2, dim=1) for _ in range(2)]
    repeatability = [torch.sigmoid(torch.randn(B, 1, H, W)) for _ in range(2)]
    reliability = [torch.sigmoid(torch.randn(B, 1, H, W)) for _ in range(2)]
    aflow = identity_aflow(B, H, W)
    mask = torch.ones(B, H, W, dtype=torch.uint8)
    return dict(descriptors=descriptors, repeatability=repeatability,
                reliability=reliability, aflow=aflow, mask=mask)


@pytest.fixture(scope="module")
def standard_loss():
    """MultiLoss with the same hyper-parameters used in train.py."""
    sampler = NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5,
                          border=16, subd_neg=-8, maxpool_pos=True)
    return MultiLoss(
        1, ReliabilityLoss(sampler, base=0.5, nq=20),
        1, CosimLoss(N=16),
        1, PeakyLoss(N=16),
    )


# ── Quad_L2Net_ConfCFS ────────────────────────────────────────────────────────

class TestQuadL2NetConfCFS:
    """
    Tests for the primary R2D2-style network used in training.
    All tests run on CPU with small synthetic inputs so they finish quickly.
    """

    @pytest.fixture
    def model(self):
        return Quad_L2Net_ConfCFS().eval()

    def _forward(self, model, B=1, H=48, W=48):
        imgs = [torch.randn(B, 3, H, W) for _ in range(2)]
        with torch.no_grad():
            return model(imgs=imgs)

    # ── output keys

    def test_output_contains_required_keys(self, model):
        out = self._forward(model)
        assert "descriptors" in out
        assert "reliability" in out
        assert "repeatability" in out

    # ── spatial resolution

    def test_descriptor_spatial_size_matches_input(self, model):
        # Dilated-conv design — output resolution must equal input resolution.
        for H, W in [(32, 32), (48, 64), (96, 128)]:
            out = self._forward(model, H=H, W=W)
            for desc in out["descriptors"]:
                assert desc.shape[-2:] == (H, W), \
                    f"Expected ({H},{W}), got {desc.shape[-2:]}"

    # ── descriptor normalisation (idea 2)

    def test_descriptors_are_unit_l2_norm(self, model):
        out = self._forward(model, B=2)
        for desc in out["descriptors"]:   # desc: (B, 128, H, W)
            norms = desc.norm(p=2, dim=1)
            assert norms.allclose(torch.ones_like(norms), atol=1e-5), \
                f"Descriptor norms not 1; min={norms.min():.4f} max={norms.max():.4f}"

    # ── confidence ranges (idea 3)

    def test_reliability_in_unit_interval(self, model):
        out = self._forward(model, B=2)
        for rel in out["reliability"]:
            assert rel.min().item() >= 0.0
            assert rel.max().item() <= 1.0

    def test_repeatability_in_unit_interval(self, model):
        out = self._forward(model, B=2)
        for rep in out["repeatability"]:
            assert rep.min().item() >= 0.0
            assert rep.max().item() <= 1.0

    def test_gradients_populate_all_parameters(self, model):
        # Backprop through the mean of every output tensor; every learnable
        # parameter must receive a non-None gradient.
        model.train()
        imgs = [torch.randn(1, 3, 48, 48), torch.randn(1, 3, 48, 48)]
        outputs = model(imgs=imgs)
        loss = sum(
            t.mean()
            for key in ("descriptors", "reliability", "repeatability")
            for t in outputs[key]
        )
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for: {name}"


# ── ConvNeXtV2 ────────────────────────────────────────────────────────────────

class TestConvNeXtV2:
    """
    Tests for the ConvNeXtV2 alternative backbone.
    Uses 'nano' scale (fewest parameters) to keep tests fast.
    """

    @pytest.fixture
    def model(self):
        return ConvNeXtV2(model_scale="nano").eval()

    def _forward(self, model, B=1, H=32, W=32):
        imgs = [torch.randn(B, 3, H, W) for _ in range(2)]
        with torch.no_grad():
            return model(imgs=imgs)

    def test_output_contains_required_keys(self, model):
        out = self._forward(model)
        assert {"descriptors", "reliability", "repeatability"} <= out.keys()

    def test_descriptors_are_unit_l2_norm(self, model):
        out = self._forward(model, B=2)
        for desc in out["descriptors"]:
            norms = desc.norm(p=2, dim=1)
            assert norms.allclose(torch.ones_like(norms), atol=1e-5)

    def test_resolution_preserved_by_dilated_design(self, model):
        # Stages use dilation instead of striding — no downsampling ever occurs.
        for H, W in [(32, 32), (48, 64)]:
            out = self._forward(model, H=H, W=W)
            assert out["descriptors"][0].shape[-2:] == (H, W), \
                f"Expected ({H},{W}), got {out['descriptors'][0].shape[-2:]}"

    def test_gradients_populate_all_parameters(self, model):
        model.train()
        imgs = [torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)]
        outputs = model(imgs=imgs)
        loss = sum(
            t.mean()
            for key in ("descriptors", "reliability", "repeatability")
            for t in outputs[key]
        )
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for: {name}"


# ── MultiLoss ─────────────────────────────────────────────────────────────────

class TestMultiLoss:
    def test_loss_is_finite_scalar(self, standard_loss):
        total, _ = standard_loss(**random_batch())
        assert total.ndim == 0
        assert torch.isfinite(total)

    def test_loss_details_contain_aggregate_key(self, standard_loss):
        _, details = standard_loss(**random_batch())
        assert "loss" in details

    def test_loss_produces_gradients_for_inputs(self, standard_loss):
        # Verify the loss differentiates w.r.t. its direct inputs (descriptors,
        # repeatability, reliability) independently of any model.
        # Inputs are leaf tensors so .grad is populated by backward().
        B, D, H, W = 1, 128, 48, 48
        descriptors = [
            F.normalize(torch.randn(B, D, H, W), p=2, dim=1).detach().requires_grad_(True)
            for _ in range(2)
        ]
        repeatability = [
            torch.sigmoid(torch.randn(B, 1, H, W)).detach().requires_grad_(True)
            for _ in range(2)
        ]
        reliability = [
            torch.sigmoid(torch.randn(B, 1, H, W)).detach().requires_grad_(True)
            for _ in range(2)
        ]
        total, _ = standard_loss(
            descriptors=descriptors,
            repeatability=repeatability,
            reliability=reliability,
            aflow=identity_aflow(B, H, W),
            mask=torch.ones(B, H, W, dtype=torch.uint8),
        )
        total.backward()
        for i, (d, rep, rel) in enumerate(zip(descriptors, repeatability, reliability)):
            assert d.grad is not None,   f"No gradient for descriptors[{i}]"
            assert rep.grad is not None, f"No gradient for repeatability[{i}]"
            assert rel.grad is not None, f"No gradient for reliability[{i}]"


# ── PeakyLoss ─────────────────────────────────────────────────────────────────

class TestPeakyLoss:
    """idea 6: verify the loss pushes saliency maps toward peakiness."""

    @pytest.fixture
    def loss_fn(self):
        return PeakyLoss(N=16)

    def test_uniform_saliency_gives_high_loss(self, loss_fn):
        # A uniform map has no peaks (max ≈ avg), so peaky loss should be high.
        # Border zero-padding from the internal AvgPool prevents it from being
        # exactly 1.0, but it should stay well above 0.8.
        uniform = torch.full((1, 1, 48, 48), 0.5)
        loss = loss_fn(repeatability=[uniform, uniform])
        assert loss.item() > 0.8

    def test_random_saliency_gives_lower_loss_than_uniform(self, loss_fn):
        # A random map has many local maxima (max > avg in most windows), so
        # (maxpool − avgpool).mean() is large → 1 − that → lower loss.
        # A constant map has max == avg everywhere (ignoring borders) → high loss.
        torch.manual_seed(0)
        uniform = torch.full((1, 1, 48, 48), 0.5)
        random_map = torch.rand(1, 1, 48, 48)
        assert loss_fn(repeatability=[random_map, random_map]).item() \
             < loss_fn(repeatability=[uniform, uniform]).item()


# ── NghSampler2 ───────────────────────────────────────────────────────────────

class TestNghSampler2:
    """idea 7: verify positive/negative labelling is correct."""

    @pytest.fixture
    def sampler(self):
        return NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5,
                           border=16, subd_neg=-8, maxpool_pos=True)

    @pytest.fixture
    def inputs(self):
        B, D, H, W = 2, 128, 48, 48
        feat = F.normalize(torch.randn(B, D, H, W), p=2, dim=1)
        conf = torch.ones(B, 1, H, W) * 0.9
        aflow = identity_aflow(B, H, W)
        return feat, feat.clone(), conf, conf.clone(), aflow

    def test_first_column_is_always_positive(self, sampler, inputs):
        feat1, feat2, conf1, conf2, aflow = inputs
        # With identity aflow feat1 == feat2 at every corresponding position →
        # positive score = 1.0, which maxpool always selects.
        _, gt, _, _ = sampler([feat1, feat2], [conf1, conf2], aflow)
        assert gt[:, 0].all(), "First gt column must be all-ones (positive)"

    def test_no_other_positive_columns_with_maxpool(self, sampler, inputs):
        feat1, feat2, conf1, conf2, aflow = inputs
        _, gt, _, _ = sampler([feat1, feat2], [conf1, conf2], aflow)
        # With maxpool_pos=True, pscores collapses to shape (N, 1), so only
        # the first gt column is labelled positive.
        assert gt[:, 1:].sum().item() == 0

    def test_scores_and_gt_shapes_are_consistent(self, sampler, inputs):
        feat1, feat2, conf1, conf2, aflow = inputs
        scores, gt, mask, _ = sampler([feat1, feat2], [conf1, conf2], aflow)
        assert scores.shape == gt.shape


# ── NonMaxSuppression ─────────────────────────────────────────────────────────

class TestNonMaxSuppression:
    """idea 8: NMS should select only true local maxima above the threshold."""

    def test_single_peak_selected_at_correct_location(self):
        nms = NonMaxSuppression(rel_thr=0.5, rep_thr=0.5)
        H, W = 16, 16
        rep = torch.zeros(1, 1, H, W)
        rel = torch.ones(1, 1, H, W)
        rep[0, 0, 5, 7] = 1.0   # peak at (y=5, x=7)

        # NMS returns (2, N) where row 0 = y, row 1 = x
        result = nms(reliability=[rel], repeatability=[rep])
        assert result.shape[1] == 1, "Exactly one keypoint expected"
        assert (result[0, 0].item(), result[1, 0].item()) == (5, 7)

    def test_pixels_below_threshold_are_suppressed(self):
        nms = NonMaxSuppression(rel_thr=0.9, rep_thr=0.9)
        rep = torch.full((1, 1, 16, 16), 0.5)   # all below rep_thr=0.9
        rel = torch.ones(1, 1, 16, 16)
        result = nms(reliability=[rel], repeatability=[rep])
        assert result.shape[1] == 0, "No keypoints should pass the threshold"

    def test_two_distant_peaks_both_selected(self):
        nms = NonMaxSuppression(rel_thr=0.5, rep_thr=0.5)
        rep = torch.zeros(1, 1, 32, 32)
        rel = torch.ones(1, 1, 32, 32)
        rep[0, 0, 5, 5] = 1.0    # peak A
        rep[0, 0, 20, 20] = 1.0  # peak B — > 3 px away, so 3×3 max-filter
        #                           does not merge the two maxima
        result = nms(reliability=[rel], repeatability=[rep])
        assert result.shape[1] == 2, "Both distant peaks should be detected"

    def test_neighbours_of_peak_are_not_selected(self):
        # With a threshold above zero, zero-valued neighbour pixels fail the
        # rep_thr gate even though the 3×3 max-filter gives them the peak value.
        nms = NonMaxSuppression(rel_thr=0.5, rep_thr=0.5)
        rep = torch.zeros(1, 1, 16, 16)
        rel = torch.ones(1, 1, 16, 16)
        rep[0, 0, 8, 8] = 1.0   # single maximum above threshold
        result = nms(reliability=[rel], repeatability=[rep])
        assert result.shape[1] == 1, "Only the peak itself should be selected"
