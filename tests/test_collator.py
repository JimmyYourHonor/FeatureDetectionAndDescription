"""
Tests for parametric_collator — the data collator wired into TrainingArguments
that handles both ParametricTransform output and HPatches eval batches.
"""

import torch

from preprocessing.collator import parametric_collator


# ── ParametricTransform output ────────────────────────────────────────────────

class TestParametricBatches:
    def _sample(self, src_size=(64, 64), flow_size=(1, 1)):
        H, W = src_size
        Hf, Wf = flow_size
        return {
            'src_a': torch.zeros(3, H, W, dtype=torch.uint8),
            'src_b': torch.zeros(3, H, W, dtype=torch.uint8),
            'sa2ia': torch.eye(3, dtype=torch.float32),
            'sb2ib': torch.eye(3, dtype=torch.float32),
            'M_ab': torch.eye(3, dtype=torch.float32),
            'img_size': torch.tensor([W, H, W, H], dtype=torch.int32),
            'mode': 0,
            'aflow_full': torch.zeros(2, Hf, Wf, dtype=torch.float32),
            'mask_full': torch.zeros(Hf, Wf, dtype=torch.uint8),
        }

    def test_variable_resolution_keys_are_kept_as_lists(self):
        # src_a/src_b plus aflow_full/mask_full all hold variable per-sample
        # resolutions and must not be stacked.
        batch = parametric_collator([self._sample(), self._sample()])
        for k in ('src_a', 'src_b', 'aflow_full', 'mask_full'):
            assert isinstance(batch[k], list), f"{k} should be a list"
            assert len(batch[k]) == 2

    def test_fixed_size_keys_are_stacked(self):
        batch = parametric_collator([self._sample(), self._sample()])
        assert batch['sa2ia'].shape == (2, 3, 3)
        assert batch['sb2ib'].shape == (2, 3, 3)
        assert batch['M_ab'].shape == (2, 3, 3)
        assert batch['img_size'].shape == (2, 4)

    def test_mode_is_kept_as_list_of_ints(self):
        batch = parametric_collator([self._sample(), self._sample()])
        assert isinstance(batch['mode'], list)
        assert batch['mode'] == [0, 0]

    def test_variable_resolution_sources_dont_stack(self):
        s1 = self._sample(src_size=(64, 80), flow_size=(1, 1))
        s2 = self._sample(src_size=(96, 32), flow_size=(64, 80))
        batch = parametric_collator([s1, s2])
        assert batch['src_a'][0].shape == (3, 64, 80)
        assert batch['src_a'][1].shape == (3, 96, 32)
        assert batch['aflow_full'][0].shape == (2, 1, 1)
        assert batch['aflow_full'][1].shape == (2, 64, 80)


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
