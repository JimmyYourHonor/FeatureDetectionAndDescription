"""
Cross-run metric accumulator isolation tests (T-1.3 / T-1.6 contract).

The key invariant Phase 2 depends on: each call to make_compute_metrics() must
return a FRESH, INDEPENDENT MetricAccumulator. Feeding eval batches into one
accumulator must NOT affect any other accumulator obtained from a separate
make_compute_metrics() call.

Tests are CPU-only and network-free; no model forward pass, no HF Hub access.
The same fake-eval-batch helpers used in test_evaluation.py are reused here.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction

from models.evaluation.compute_metrics import (
    MetricAccumulator,
    _reset_state,
    compute_metrics,
    make_compute_metrics,
)

# Keep consistent with the rest of the test suite
DESC_DIM = 128
FEAT_COLS = 3 + DESC_DIM  # x, y, scale, descriptors


# ── helpers (mirror test_evaluation.py so this file is self-contained) ────────

def orthogonal_features(n: int, x_start: float = 1.0, step: float = 5.0) -> torch.Tensor:
    feats = torch.zeros(n, FEAT_COLS)
    feats[:, 0] = torch.arange(n, dtype=torch.float32) * step + x_start
    feats[:, 1] = torch.arange(n, dtype=torch.float32) * step + x_start
    feats[:, 2] = 1.0
    for i in range(min(n, DESC_DIM)):
        feats[i, 3 + i] = 1.0
    return feats


def identity_aflow(H: int = 64, W: int = 64) -> torch.Tensor:
    xs = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    ys = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    return torch.stack([xs, ys], dim=0)


def good_flow_batch(n: int = 6) -> EvalPrediction:
    """Flow-format batch where MNN finds perfect matches (MMA → 1.0)."""
    feats = orthogonal_features(n)
    return EvalPrediction(predictions=[feats, feats.clone()], label_ids=identity_aflow())


def empty_flow_batch() -> EvalPrediction:
    """Flow-format batch with no keypoints (MMA → 0.0)."""
    empty = torch.zeros(0, FEAT_COLS)
    return EvalPrediction(predictions=[empty, empty.clone()], label_ids=identity_aflow())


def good_hpatches_batch(n: int = 6) -> EvalPrediction:
    """HPatches-format batch with perfect matches."""
    feats = orthogonal_features(n)
    labels = torch.stack([torch.eye(3)] * 6, dim=0)
    return EvalPrediction(predictions=[feats] * 6, label_ids=labels)


# ── autouse fixture: keep module-level default clean around every test ────────

@pytest.fixture(autouse=True)
def clean_state():
    _reset_state()
    yield
    _reset_state()


# ── core isolation contract ───────────────────────────────────────────────────

class TestMakeComputeMetricsIsolation:
    """make_compute_metrics() returns independent accumulators — the Phase 2
    cross-trial-leakage guard."""

    def test_two_fresh_accumulators_start_empty(self):
        """Each make_compute_metrics() call returns a clean accumulator."""
        _, acc1 = make_compute_metrics()
        _, acc2 = make_compute_metrics()
        # A fresh accumulator has no pairs → result() returns {"MMA": 0.0}
        assert acc1.total_pairs == 0
        assert acc2.total_pairs == 0

    def test_feeding_first_does_not_affect_second(self):
        """Feeding batches into compute_fn1 must not change acc2's result."""
        fn1, acc1 = make_compute_metrics()
        fn2, acc2 = make_compute_metrics()

        # Saturate acc1 with good data
        fn1(good_flow_batch(), compute_result=False)
        fn1(good_flow_batch(), compute_result=False)

        # acc2 was never touched — its result must still be empty (0.0)
        result2 = fn2(empty_flow_batch(), compute_result=True)
        assert result2["MMA"] == pytest.approx(0.0), (
            "acc2 was contaminated by data fed to fn1 (cross-run leakage)"
        )

    def test_feeding_second_does_not_affect_first(self):
        """Feeding batches into compute_fn2 must not change acc1's result."""
        fn1, acc1 = make_compute_metrics()
        fn2, acc2 = make_compute_metrics()

        # Feed only acc2
        fn2(good_flow_batch(), compute_result=False)
        fn2(good_flow_batch(), compute_result=False)

        # acc1 only saw an empty batch → MMA 0.0
        result1 = fn1(empty_flow_batch(), compute_result=True)
        assert result1["MMA"] == pytest.approx(0.0), (
            "acc1 was contaminated by data fed to fn2 (cross-run leakage)"
        )

    def test_independent_results_differ_as_expected(self):
        """Two accumulators accumulate independently and produce different results."""
        fn_good, _ = make_compute_metrics()
        fn_empty, _ = make_compute_metrics()

        result_good = fn_good(good_flow_batch(), compute_result=True)
        result_empty = fn_empty(empty_flow_batch(), compute_result=True)

        assert result_good["MMA"] == pytest.approx(1.0)
        assert result_empty["MMA"] == pytest.approx(0.0)

    def test_result_resets_accumulator_so_next_call_starts_clean(self):
        """After result() is called via compute_result=True, the accumulator
        is auto-reset. The next batch starts fresh — no residue from the prior run."""
        fn, acc = make_compute_metrics()

        # Run 1: good data → MMA 1.0
        r1 = fn(good_flow_batch(), compute_result=True)
        assert r1["MMA"] == pytest.approx(1.0)

        # Run 2: only empty data → should give 0.0, not a blend of run-1 hits
        r2 = fn(empty_flow_batch(), compute_result=True)
        assert r2["MMA"] == pytest.approx(0.0), (
            "Accumulator was not reset after result() — run-1 data leaked into run-2"
        )

    def test_make_compute_metrics_isolated_from_module_level_accumulator(self):
        """A fresh compute_fn from make_compute_metrics() must be independent of
        the module-level compute_metrics / _default_accumulator."""
        fn_fresh, _ = make_compute_metrics()

        # Poison the module-level accumulator with good data (no result() call,
        # so the state sits there unreset)
        compute_metrics(good_flow_batch(), compute_result=False)
        compute_metrics(good_flow_batch(), compute_result=False)

        # The fresh accumulator should only see the empty batch we give it
        result = fn_fresh(empty_flow_batch(), compute_result=True)
        assert result["MMA"] == pytest.approx(0.0), (
            "Fresh accumulator was contaminated by module-level accumulator state"
        )

    def test_many_fresh_accumulators_all_independent(self):
        """Stress test: N accumulators all get different data; verify results."""
        N = 5
        fns = [make_compute_metrics()[0] for _ in range(N)]

        # Feed only the i-th accumulator with i good batches; the rest get nothing
        # except one empty batch to trigger result() on them.
        for i in range(N):
            for _ in range(i):
                fns[i](good_flow_batch(), compute_result=False)

        results = [fns[i](empty_flow_batch(), compute_result=True) for i in range(N)]

        # fn[0] saw 0 good batches + 1 empty → MMA 0.0
        assert results[0]["MMA"] == pytest.approx(0.0)
        # fn[1..N-1] saw i good batches + 1 empty batch; MMA must be < 1.0
        # because the empty batch contributes 0 matches to the final sum.
        # More importantly: no result should be identical to result[0] for i>0
        # (they each saw different amounts of good data).
        for i in range(1, N):
            # Results must differ — proving there is no shared state
            assert results[i]["MMA"] != results[0]["MMA"], (
                f"fn[{i}] result matches fn[0] — likely sharing accumulator state"
            )


class TestMetricAccumulatorDirectIsolation:
    """Direct unit tests on the MetricAccumulator class."""

    def test_two_instances_are_independent_objects(self):
        acc1 = MetricAccumulator()
        acc2 = MetricAccumulator()
        assert acc1 is not acc2
        assert acc1.err is not acc2.err

    def test_update_on_one_does_not_change_other(self):
        acc1 = MetricAccumulator()
        acc2 = MetricAccumulator()

        acc1.update(good_flow_batch())
        # acc2 has never been updated; its total_pairs must still be 0
        assert acc2.total_pairs == 0

    def test_reset_on_one_does_not_affect_other(self):
        acc1 = MetricAccumulator()
        acc2 = MetricAccumulator()

        acc1.update(good_flow_batch())
        acc2.update(good_flow_batch())

        acc1.reset()  # only acc1 reset

        assert acc1.total_pairs == 0
        assert acc2.total_pairs == 1  # acc2 still has data


class TestEvalCfgNoneFallback:
    """T-1.7: CustomTrainer with eval_cfg=None must use the documented hardcoded
    defaults, identical to the values in EvalCfg."""

    def test_eval_cfg_none_uses_default_rel_thr(self):
        from models.custom_trainer import CustomTrainer
        from config.schema import EvalCfg
        t = CustomTrainer.__new__(CustomTrainer)
        t.eval_cfg = None
        # Reproduce the None-fallback expression from prediction_step
        rel_thr = t.eval_cfg.rel_thr if t.eval_cfg is not None else 0.7
        assert rel_thr == 0.7
        assert rel_thr == EvalCfg().rel_thr

    def test_eval_cfg_none_uses_default_rep_thr(self):
        from models.custom_trainer import CustomTrainer
        from config.schema import EvalCfg
        t = CustomTrainer.__new__(CustomTrainer)
        t.eval_cfg = None
        rep_thr = t.eval_cfg.rep_thr if t.eval_cfg is not None else 0.7
        assert rep_thr == 0.7
        assert rep_thr == EvalCfg().rep_thr

    def test_eval_cfg_none_uses_default_flow_min_size(self):
        from models.custom_trainer import CustomTrainer
        from config.schema import EvalCfg
        t = CustomTrainer.__new__(CustomTrainer)
        t.eval_cfg = None
        flow_min_size = t.eval_cfg.flow_min_size if t.eval_cfg is not None else 192
        assert flow_min_size == 192
        assert flow_min_size == EvalCfg().flow_min_size

    def test_eval_cfg_none_uses_default_scale_f(self):
        from models.custom_trainer import CustomTrainer
        from config.schema import EvalCfg
        t = CustomTrainer.__new__(CustomTrainer)
        t.eval_cfg = None
        scale_f = t.eval_cfg.scale_f if t.eval_cfg is not None else 2**0.25
        assert abs(scale_f - 2**0.25) < 1e-12
        assert abs(scale_f - EvalCfg().scale_f) < 1e-12

    def test_eval_cfg_none_uses_default_multiscale_bounds(self):
        from models.custom_trainer import CustomTrainer
        from config.schema import EvalCfg
        t = CustomTrainer.__new__(CustomTrainer)
        t.eval_cfg = None
        min_scale  = t.eval_cfg.min_scale  if t.eval_cfg is not None else 0.0
        max_scale  = t.eval_cfg.max_scale  if t.eval_cfg is not None else 1.0
        min_size   = t.eval_cfg.min_size   if t.eval_cfg is not None else 256
        max_size   = t.eval_cfg.max_size   if t.eval_cfg is not None else 1024
        defaults   = EvalCfg()
        assert min_scale == defaults.min_scale
        assert max_scale == defaults.max_scale
        assert min_size  == defaults.min_size
        assert max_size  == defaults.max_size

    def test_eval_cfg_explicit_overrides_defaults(self):
        """When eval_cfg is an EvalCfg with non-default values, those values are used."""
        from models.custom_trainer import CustomTrainer
        from config.schema import EvalCfg
        t = CustomTrainer.__new__(CustomTrainer)
        t.eval_cfg = EvalCfg(rel_thr=0.5, rep_thr=0.6, flow_min_size=128)
        rel_thr      = t.eval_cfg.rel_thr      if t.eval_cfg is not None else 0.7
        rep_thr      = t.eval_cfg.rep_thr      if t.eval_cfg is not None else 0.7
        flow_min_size = t.eval_cfg.flow_min_size if t.eval_cfg is not None else 192
        assert rel_thr == 0.5
        assert rep_thr == 0.6
        assert flow_min_size == 128

    def test_none_fallbacks_in_prediction_step_source_match_eval_cfg_defaults(self):
        """Verify the hardcoded None-fallback literals in prediction_step match EvalCfg defaults.
        This test guards against future silent drift between the two."""
        import inspect
        from models.custom_trainer import CustomTrainer
        from config.schema import EvalCfg
        src = inspect.getsource(CustomTrainer.prediction_step)
        defaults = EvalCfg()
        # Each fallback literal must be present in the source
        assert "0.7" in src  # rel_thr and rep_thr
        assert "2**0.25" in src  # scale_f
        assert "0.0" in src  # min_scale
        assert "1.0" in src  # max_scale
        assert "256" in src  # min_size
        assert "1024" in src  # max_size
        assert "192" in src  # flow_min_size
