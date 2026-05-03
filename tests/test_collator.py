"""
Tests for parametric_collator — the data collator wired into TrainingArguments
that handles both ParametricTransform output and HPatches eval batches.
"""

import pytest
import torch

from preprocessing.collator import parametric_collator


# ── ParametricTransform output ────────────────────────────────────────────────

class TestParametricBatches:
    def _sample(self, src_size=(64, 64)):
        H, W = src_size
        return {
            'src_a': torch.zeros(3, H, W, dtype=torch.uint8),
            'src_b': torch.zeros(3, H, W, dtype=torch.uint8),
            'M_a': torch.eye(3, dtype=torch.float32),
            'M_b': torch.eye(3, dtype=torch.float32),
            'aflow': torch.zeros(2, 16, 16, dtype=torch.float32),
            'mask': torch.zeros(16, 16, dtype=torch.uint8),
        }

    def test_src_keys_are_kept_as_lists(self):
        batch = parametric_collator([self._sample(), self._sample()])
        assert isinstance(batch['src_a'], list)
        assert isinstance(batch['src_b'], list)
        assert len(batch['src_a']) == 2

    def test_fixed_size_keys_are_stacked(self):
        batch = parametric_collator([self._sample(), self._sample()])
        assert batch['M_a'].shape == (2, 3, 3)
        assert batch['M_b'].shape == (2, 3, 3)
        assert batch['aflow'].shape == (2, 2, 16, 16)
        assert batch['mask'].shape == (2, 16, 16)

    def test_variable_resolution_sources_dont_stack(self):
        s1 = self._sample(src_size=(64, 80))
        s2 = self._sample(src_size=(96, 32))
        batch = parametric_collator([s1, s2])
        assert batch['src_a'][0].shape == (3, 64, 80)
        assert batch['src_a'][1].shape == (3, 96, 32)


# ── HPatches eval format ──────────────────────────────────────────────────────

class TestHPatchesBatches:
    def _hpatches_sample(self):
        sample = {f'{i}.ppm': torch.zeros(3, 32, 32) for i in range(1, 7)}
        for i in range(2, 7):
            sample[f'h_1_{i}'] = torch.eye(3, dtype=torch.float32)
        return sample

    def test_hpatches_default_collation_via_stack(self):
        # No src_a/src_b keys → every tensor stacks into the batch dim.
        batch = parametric_collator([self._hpatches_sample()])
        assert batch['1.ppm'].shape == (1, 3, 32, 32)
        assert batch['h_1_2'].shape == (1, 3, 3)
