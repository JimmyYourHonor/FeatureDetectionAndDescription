"""
Tests for GPUBatchAugment — the batched augmentation+normalization step that
runs after GPUWarp's geometric warp on the GPU.

These tests run on CPU (no CUDA required); the module is dtype/device-agnostic.
"""

import pytest
import torch

from preprocessing.gpu_augment import GPUBatchAugment


# ── helpers ───────────────────────────────────────────────────────────────────

def _uint8_batch(B=2, C=3, H=32, W=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, C, H, W), generator=g, dtype=torch.uint8)


# ── shape / dtype contract ────────────────────────────────────────────────────

class TestForwardContract:
    def test_output_shape_matches_input(self):
        aug = GPUBatchAugment()
        a, b = _uint8_batch(), _uint8_batch(seed=1)
        out_a, out_b = aug(a, b, training=True)
        assert out_a.shape == a.shape
        assert out_b.shape == b.shape

    def test_output_is_float32(self):
        aug = GPUBatchAugment()
        out_a, out_b = aug(_uint8_batch(), _uint8_batch(seed=1))
        assert out_a.dtype == torch.float32
        assert out_b.dtype == torch.float32

    def test_accepts_float_input(self):
        aug = GPUBatchAugment()
        a = torch.rand(2, 3, 16, 16)
        b = torch.rand(2, 3, 16, 16)
        out_a, out_b = aug(a, b)
        assert out_a.shape == a.shape


# ── img_a is never augmented ──────────────────────────────────────────────────

class TestImgAIsNotAugmented:
    def test_img_a_is_only_normalized(self):
        # img_a should equal (input/255 - mean) / std exactly — no jitter / noise
        aug = GPUBatchAugment()
        a = _uint8_batch(seed=42)
        b = _uint8_batch(seed=43)
        out_a, _ = aug(a, b, training=True)

        expected = a.float() / 255.0
        expected = (expected - aug._rgb_mean) / aug._rgb_std
        torch.testing.assert_close(out_a, expected)


# ── eval mode skips noise + color jitter ──────────────────────────────────────

class TestEvalMode:
    def test_eval_is_deterministic(self):
        # With training=False, two calls on the same input give identical results.
        aug = GPUBatchAugment()
        b = _uint8_batch(seed=7)
        out1 = aug(b, b, training=False)
        out2 = aug(b, b, training=False)
        torch.testing.assert_close(out1[1], out2[1])

    def test_eval_only_normalizes_img_b(self):
        aug = GPUBatchAugment()
        b = _uint8_batch(seed=11)
        _, out_b = aug(b, b, training=False)
        expected = b.float() / 255.0
        expected = (expected - aug._rgb_mean) / aug._rgb_std
        torch.testing.assert_close(out_b, expected)


# ── normalization values ──────────────────────────────────────────────────────

class TestNormalization:
    def test_zero_input_maps_to_negative_mean_over_std(self):
        # An all-zero image, normalized, should equal -mean/std per channel.
        aug = GPUBatchAugment()
        zeros = torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
        out_a, _ = aug(zeros, zeros, training=False)
        expected_per_channel = -aug._rgb_mean / aug._rgb_std
        # (1, 3, 1, 1) broadcast vs (1, 3, 8, 8): all pixels per channel equal
        torch.testing.assert_close(out_a, expected_per_channel.expand_as(out_a))


# ── pixel noise amplitude bound ───────────────────────────────────────────────

class TestPixelNoise:
    def test_noise_magnitude_within_bound(self):
        # With color jitter disabled, |aug_b - normalize(b)| ≤ noise_ampl/255 / std
        # per channel (uniform noise added in [0,1] space then normalized).
        aug = GPUBatchAugment(brightness=0, contrast=0, saturation=0, hue=0,
                              noise_ampl=25.0)
        b = _uint8_batch(seed=3)
        b_norm = (b.float() / 255.0 - aug._rgb_mean) / aug._rgb_std
        _, out_b = aug(b, b, training=True)

        max_delta_per_channel = (25.0 / 2 / 255.0) / aug._rgb_std  # (1,3,1,1)
        delta = (out_b - b_norm).abs()
        # Allow tiny slack for clamping at 0/1 boundaries which can change delta.
        # The bound should hold for non-saturated pixels; check the median.
        per_chan_max = delta.amax(dim=(0, 2, 3), keepdim=True)
        # Either within bound, or the input was near 0/1 (clamp absorbs it).
        # Use a relaxed bound: 1.05x to allow float roundoff.
        assert (per_chan_max <= max_delta_per_channel.squeeze(0) * 1.05 + 1e-5).all()


# ── color jitter is per-sample ────────────────────────────────────────────────

class TestColorJitterIsPerSample:
    def test_different_samples_get_different_factors(self):
        # Feed identical content for two samples and verify the augmented
        # outputs differ across the batch dimension (different random factors).
        aug = GPUBatchAugment(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0,
                              noise_ampl=0.0)
        torch.manual_seed(0)
        same = torch.full((2, 3, 16, 16), 128, dtype=torch.uint8)
        _, out_b = aug(same, same, training=True)
        # The two batch elements should have been augmented independently.
        assert not torch.allclose(out_b[0], out_b[1])


# ── HSV roundtrip used by hue ─────────────────────────────────────────────────

class TestHSVRoundtrip:
    def test_zero_hue_shift_recovers_image(self):
        # _adjust_hue with shift=0 should recover the original (within float eps).
        aug = GPUBatchAugment()
        torch.manual_seed(0)
        img = torch.rand(2, 3, 8, 8)
        shift = torch.zeros(2, 1, 1, 1)
        out = aug._adjust_hue(img, shift)
        torch.testing.assert_close(out, img.clamp(0, 1), atol=1e-5, rtol=1e-4)
