"""Unit tests for Phase 2 Optuna search mechanisms (T-2.1, T-2.2, T-2.3).

All tests are CPU-only and network-free. No training is performed.
WandB is already mocked by conftest.py.

Covered:
  - sample_space: forces num_train_epochs==3, sets dotted-paths, invariant violation
  - Space-spec type handlers: float/log, int, categorical each call the right suggest_*
  - OptunaPruningCallback.on_evaluate: reports and raises when should_prune; skips
    when eval_flow_MMA is absent from metrics
  - StudyResult best_cfg reconstruction via FixedTrial -> sample_space
"""

import sys
import os

import pytest
import optuna

# Silence optuna logging during tests
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.schema import RunCfg
from optuna_search import (
    DEFAULT_SPACE_SPEC,
    StudyResult,
    _assert_sampler_invariants,
    _set_dotted,
    sample_space,
)
from constants import FLOW_MMA_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_cfg() -> RunCfg:
    """Return a default RunCfg so tests don't depend on HF Hub."""
    return RunCfg()


def _fixed_trial(params: dict) -> optuna.trial.FixedTrial:
    """Construct an optuna FixedTrial with the given param dict."""
    return optuna.trial.FixedTrial(params)


def _valid_default_params() -> dict:
    """A params dict where every DEFAULT_SPACE_SPEC path has a valid value.

    Chosen so that the sampler invariant pos_d < neg_d <= ngh holds:
      pos_d=1, neg_d=4, ngh=8
    """
    return {
        "optim.learning_rate": 1e-3,
        "optim.weight_decay": 1e-4,
        "optim.warmup_steps": 300,
        "sampler.ngh": 8,
        "sampler.pos_d": 1,
        "sampler.neg_d": 4,
        "loss.reliability.weight": 1.0,
        "loss.cosim.weight": 1.0,
        "loss.peaky.weight": 1.0,
        "augment.brightness": 0.2,
        "augment.contrast": 0.2,
        "augment.saturation": 0.2,
        "augment.hue": 0.1,
        "augment.noise_ampl": 25.0,
        "eval.rel_thr": 0.7,
        "eval.rep_thr": 0.7,
    }


# ---------------------------------------------------------------------------
# T-2.2: sample_space contract
# ---------------------------------------------------------------------------

class TestSampleSpaceEpochForcing:
    """sample_space must always force num_train_epochs == 3."""

    def test_forces_three_epochs_from_default_cfg(self):
        """Even if base_cfg has 25 epochs, sample_space forces 3."""
        cfg = _fresh_cfg()
        assert cfg.trainer.num_train_epochs != 3  # confirm default is 25
        params = _valid_default_params()
        out = sample_space(_fixed_trial(params), cfg, DEFAULT_SPACE_SPEC)
        assert out.trainer.num_train_epochs == 3, (
            "sample_space must force num_train_epochs=3 unconditionally"
        )

    def test_forces_three_epochs_even_if_space_spec_is_empty(self):
        """An empty space_spec still applies the epoch-3 override."""
        cfg = _fresh_cfg()
        out = sample_space(_fixed_trial({}), cfg, space_spec={})
        assert out.trainer.num_train_epochs == 3

    def test_does_not_mutate_base_cfg(self):
        """sample_space must deep-copy base_cfg; the original must be unchanged."""
        cfg = _fresh_cfg()
        original_epochs = cfg.trainer.num_train_epochs
        params = _valid_default_params()
        _ = sample_space(_fixed_trial(params), cfg, DEFAULT_SPACE_SPEC)
        assert cfg.trainer.num_train_epochs == original_epochs, (
            "sample_space mutated base_cfg — it must deep-copy before modifying"
        )


class TestSampleSpacePathAssignment:
    """Each spec'd dotted-path is set on the returned cfg to the FixedTrial value."""

    def test_float_path_assigned_correctly(self):
        params = _valid_default_params()
        params["optim.learning_rate"] = 2.5e-4
        out = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        assert out.optim.learning_rate == pytest.approx(2.5e-4)

    def test_int_path_assigned_correctly(self):
        params = _valid_default_params()
        params["optim.warmup_steps"] = 750
        out = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        assert out.optim.warmup_steps == 750

    def test_sampler_paths_all_assigned(self):
        params = _valid_default_params()
        params["sampler.ngh"] = 10
        params["sampler.pos_d"] = 2
        params["sampler.neg_d"] = 5
        out = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        assert out.sampler.ngh == 10
        assert out.sampler.pos_d == 2
        assert out.sampler.neg_d == 5

    def test_loss_weight_paths_assigned(self):
        params = _valid_default_params()
        params["loss.reliability.weight"] = 1.8
        params["loss.cosim.weight"] = 0.6
        params["loss.peaky.weight"] = 1.3
        out = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        assert out.loss.reliability.weight == pytest.approx(1.8)
        assert out.loss.cosim.weight == pytest.approx(0.6)
        assert out.loss.peaky.weight == pytest.approx(1.3)

    def test_augment_paths_assigned(self):
        params = _valid_default_params()
        params["augment.brightness"] = 0.4
        params["augment.noise_ampl"] = 10.0
        out = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        assert out.augment.brightness == pytest.approx(0.4)
        assert out.augment.noise_ampl == pytest.approx(10.0)

    def test_eval_paths_assigned(self):
        params = _valid_default_params()
        params["eval.rel_thr"] = 0.6
        params["eval.rep_thr"] = 0.8
        out = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        assert out.eval.rel_thr == pytest.approx(0.6)
        assert out.eval.rep_thr == pytest.approx(0.8)

    def test_custom_single_path_spec(self):
        """Works correctly with a minimal one-key spec."""
        spec = {"optim.weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2}}
        params = {"optim.weight_decay": 3e-4}
        out = sample_space(_fixed_trial(params), _fresh_cfg(), spec)
        assert out.optim.weight_decay == pytest.approx(3e-4)


class TestSampleSpaceInvariantEnforcement:
    """sample_space must raise ValueError for invariant-violating sampler combos."""

    def test_raises_when_pos_d_equals_neg_d(self):
        """pos_d < neg_d must hold; equal values should raise."""
        # Use a spec that allows pos_d=3, neg_d=3 (violating pos_d < neg_d).
        spec = {
            "sampler.pos_d": {"type": "int", "low": 3, "high": 3},
            "sampler.neg_d": {"type": "int", "low": 3, "high": 3},
        }
        params = {"sampler.pos_d": 3, "sampler.neg_d": 3}
        cfg = _fresh_cfg()  # default ngh=7, so neg_d=3 <= ngh=7 is fine; the violation is pos_d==neg_d
        with pytest.raises(ValueError, match="pos_d"):
            sample_space(_fixed_trial(params), cfg, spec)

    def test_raises_when_neg_d_exceeds_ngh(self):
        """neg_d <= ngh must hold."""
        spec = {
            "sampler.ngh": {"type": "int", "low": 4, "high": 4},
            "sampler.neg_d": {"type": "int", "low": 6, "high": 6},
        }
        params = {"sampler.ngh": 4, "sampler.neg_d": 6}
        cfg = _fresh_cfg()  # default pos_d=3; neg_d=6 > ngh=4 violates invariant
        with pytest.raises(ValueError, match="ngh"):
            sample_space(_fixed_trial(params), cfg, spec)

    def test_error_message_names_all_three_values(self):
        """The ValueError message should mention pos_d, neg_d, and ngh values."""
        spec = {
            "sampler.pos_d": {"type": "int", "low": 5, "high": 5},
            "sampler.neg_d": {"type": "int", "low": 5, "high": 5},
        }
        params = {"sampler.pos_d": 5, "sampler.neg_d": 5}
        cfg = _fresh_cfg()
        with pytest.raises(ValueError) as exc_info:
            sample_space(_fixed_trial(params), cfg, spec)
        msg = str(exc_info.value)
        assert "pos_d" in msg
        assert "neg_d" in msg
        assert "ngh" in msg

    def test_valid_boundary_does_not_raise(self):
        """pos_d=1, neg_d=2, ngh=2 satisfies pos_d < neg_d <= ngh exactly."""
        spec = {
            "sampler.pos_d": {"type": "int", "low": 1, "high": 1},
            "sampler.neg_d": {"type": "int", "low": 2, "high": 2},
            "sampler.ngh":   {"type": "int", "low": 2, "high": 2},
        }
        params = {"sampler.pos_d": 1, "sampler.neg_d": 2, "sampler.ngh": 2}
        cfg = _fresh_cfg()
        out = sample_space(_fixed_trial(params), cfg, spec)
        assert out.sampler.pos_d == 1
        assert out.sampler.neg_d == 2
        assert out.sampler.ngh == 2


class TestSampleSpaceTypeHandlers:
    """Each type in the spec routes to the correct suggest_* and respects constraints."""

    def test_float_without_log_is_set(self):
        spec = {"loss.reliability.weight": {"type": "float", "low": 0.5, "high": 2.0}}
        params = {"loss.reliability.weight": 1.5}
        out = sample_space(_fixed_trial(params), _fresh_cfg(), spec)
        assert out.loss.reliability.weight == pytest.approx(1.5)

    def test_float_with_log_true_is_set(self):
        spec = {"optim.learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}}
        params = {"optim.learning_rate": 3e-4}
        out = sample_space(_fixed_trial(params), _fresh_cfg(), spec)
        assert out.optim.learning_rate == pytest.approx(3e-4)

    def test_int_is_set(self):
        spec = {"optim.warmup_steps": {"type": "int", "low": 100, "high": 1000}}
        params = {"optim.warmup_steps": 600}
        out = sample_space(_fixed_trial(params), _fresh_cfg(), spec)
        assert out.optim.warmup_steps == 600

    def test_categorical_is_set(self):
        spec = {"model.name": {"type": "categorical", "choices": ["Quad_L2Net_ConfCFS", "ViTDense"]}}
        params = {"model.name": "ViTDense"}
        out = sample_space(_fixed_trial(params), _fresh_cfg(), spec)
        assert out.model.name == "ViTDense"

    def test_unsupported_type_raises_key_error(self):
        """An unsupported type string raises KeyError."""
        spec = {"optim.learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2}}
        params = {"optim.learning_rate": 3e-4}
        with pytest.raises(KeyError):
            sample_space(_fixed_trial(params), _fresh_cfg(), spec)


# ---------------------------------------------------------------------------
# T-2.2: _set_dotted helper
# ---------------------------------------------------------------------------

class TestSetDotted:
    """_set_dotted correctly navigates nested dataclasses."""

    def test_one_level_deep(self):
        cfg = RunCfg()
        _set_dotted(cfg, "model", cfg.model)  # no-op: just verifying no crash

    def test_two_levels_deep(self):
        cfg = RunCfg()
        _set_dotted(cfg, "optim.learning_rate", 9.9e-4)
        assert cfg.optim.learning_rate == pytest.approx(9.9e-4)

    def test_three_levels_deep(self):
        cfg = RunCfg()
        _set_dotted(cfg, "loss.reliability.weight", 2.5)
        assert cfg.loss.reliability.weight == pytest.approx(2.5)

    def test_invalid_intermediate_path_raises_attribute_error(self):
        """An invalid INTERMEDIATE segment (before the last) raises AttributeError
        because getattr fails on a missing nested object.

        Note: an invalid FINAL segment does NOT raise — Python dataclasses allow
        setattr with new attribute names. Validation of final-path names is deferred
        to T-3.3's validate_space_spec.
        """
        cfg = RunCfg()
        with pytest.raises(AttributeError):
            # "nonexistent_group" is an invalid intermediate segment
            _set_dotted(cfg, "nonexistent_group.learning_rate", 1.0)


# ---------------------------------------------------------------------------
# T-2.1: OptunaPruningCallback.on_evaluate
# ---------------------------------------------------------------------------

class TestOptunaPruningCallbackOnEvaluate:
    """on_evaluate reports to Optuna when flow MMA is present; skips when absent."""

    def _make_state(self, epoch: float = 1.0):
        """Minimal mock for TrainerState."""
        from unittest.mock import MagicMock
        state = MagicMock()
        state.epoch = epoch
        return state

    def test_reports_flow_mma_to_trial(self):
        """When eval_flow_MMA is in metrics, trial.report is called with it."""
        from unittest.mock import MagicMock
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = MagicMock()
        trial.should_prune.return_value = False

        cb = OptunaPruningCallback(trial)
        metrics = {FLOW_MMA_KEY: 0.65, "eval_loss": 1.2}
        state = self._make_state(epoch=1.0)

        cb.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        trial.report.assert_called_once_with(0.65, 1)

    def test_reports_at_correct_epoch_step(self):
        """report step is round(state.epoch), not global_step."""
        from unittest.mock import MagicMock
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = MagicMock()
        trial.should_prune.return_value = False

        cb = OptunaPruningCallback(trial)
        state = self._make_state(epoch=2.0)
        cb.on_evaluate(args=None, state=state, control=None, metrics={FLOW_MMA_KEY: 0.5})
        _, call_step = trial.report.call_args[0]
        assert call_step == 2

    def test_raises_trial_pruned_when_should_prune_is_true(self):
        """When trial.should_prune() returns True, raises optuna.TrialPruned."""
        from unittest.mock import MagicMock
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = MagicMock()
        trial.should_prune.return_value = True

        cb = OptunaPruningCallback(trial)
        state = self._make_state(epoch=2.0)
        with pytest.raises(optuna.TrialPruned):
            cb.on_evaluate(
                args=None,
                state=state,
                control=None,
                metrics={FLOW_MMA_KEY: 0.3},
            )

    def test_does_not_raise_when_should_prune_is_false(self):
        """When trial.should_prune() returns False, no exception is raised."""
        from unittest.mock import MagicMock
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = MagicMock()
        trial.should_prune.return_value = False

        cb = OptunaPruningCallback(trial)
        state = self._make_state(epoch=2.0)
        # Should not raise
        cb.on_evaluate(
            args=None,
            state=state,
            control=None,
            metrics={FLOW_MMA_KEY: 0.8},
        )

    def test_skips_report_when_flow_mma_key_absent(self):
        """When eval_flow_MMA is not in metrics (e.g. HPatches probe), no report."""
        from unittest.mock import MagicMock
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = MagicMock()
        trial.should_prune.return_value = True  # would prune if it got here

        cb = OptunaPruningCallback(trial)
        state = self._make_state(epoch=1.0)
        # Only HPatches key present — no eval_flow_MMA
        metrics = {"eval_MMA": 0.7, "eval_loss": 0.9}

        # Must NOT raise TrialPruned, must NOT call trial.report
        cb.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        trial.report.assert_not_called()

    def test_skips_report_for_empty_metrics(self):
        """Empty metrics dict does not crash and does not report."""
        from unittest.mock import MagicMock
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = MagicMock()
        trial.should_prune.return_value = False

        cb = OptunaPruningCallback(trial)
        state = self._make_state(epoch=1.0)
        cb.on_evaluate(args=None, state=state, control=None, metrics={})
        trial.report.assert_not_called()

    def test_uses_real_pruner_that_prunes(self):
        """Integration: use a real Optuna trial with an always-prune pruner.

        NeverPruner-inverse: we use a custom pruner that always returns True.
        Confirms end-to-end: callback raises TrialPruned when pruner says prune.
        """
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        class AlwaysPruner(optuna.pruners.BasePruner):
            def prune(self, study, trial):
                return True

        study = optuna.create_study(direction="maximize", pruner=AlwaysPruner())
        trial = study.ask()

        # Must report at least once before should_prune is meaningful
        trial.report(0.4, 1)

        cb = OptunaPruningCallback(trial)
        state = self._make_state(epoch=2.0)

        with pytest.raises(optuna.TrialPruned):
            cb.on_evaluate(
                args=None,
                state=state,
                control=None,
                metrics={FLOW_MMA_KEY: 0.3},
            )

    def test_uses_real_pruner_that_does_not_prune(self):
        """Integration: NeverPruner — callback does not raise."""
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.NopPruner(),
        )
        trial = study.ask()

        cb = OptunaPruningCallback(trial)
        state = self._make_state(epoch=1.0)

        # Should not raise
        cb.on_evaluate(
            args=None,
            state=state,
            control=None,
            metrics={FLOW_MMA_KEY: 0.8},
        )


# ---------------------------------------------------------------------------
# T-2.3: StudyResult best_cfg reconstruction via FixedTrial
# ---------------------------------------------------------------------------

class TestStudyResultBestCfgReconstruction:
    """best_cfg is recoverable from best_params via FixedTrial -> sample_space."""

    def test_round_trip_default_space(self):
        """FixedTrial(params) -> sample_space produces a RunCfg with exactly those values."""
        params = _valid_default_params()
        fixed_trial = _fixed_trial(params)
        cfg = _fresh_cfg()

        best_cfg = sample_space(fixed_trial, cfg, DEFAULT_SPACE_SPEC)

        # Spot-check several paths
        assert best_cfg.optim.learning_rate == pytest.approx(params["optim.learning_rate"])
        assert best_cfg.optim.warmup_steps == params["optim.warmup_steps"]
        assert best_cfg.sampler.ngh == params["sampler.ngh"]
        assert best_cfg.loss.reliability.weight == pytest.approx(params["loss.reliability.weight"])
        assert best_cfg.augment.brightness == pytest.approx(params["augment.brightness"])
        assert best_cfg.eval.rel_thr == pytest.approx(params["eval.rel_thr"])

    def test_round_trip_forces_three_epochs(self):
        """Reconstructed best_cfg always has num_train_epochs=3."""
        params = _valid_default_params()
        best_cfg = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        assert best_cfg.trainer.num_train_epochs == 3

    def test_study_result_dataclass_fields(self):
        """StudyResult can be constructed and its fields are accessible."""
        params = _valid_default_params()
        best_cfg = sample_space(_fixed_trial(params), _fresh_cfg(), DEFAULT_SPACE_SPEC)
        result = StudyResult(
            best_flow_MMA=0.72,
            best_params=params,
            best_cfg=best_cfg,
            study_name="test_study",
            n_trials=5,
        )
        assert result.best_flow_MMA == pytest.approx(0.72)
        assert result.best_params is params
        assert result.best_cfg is best_cfg
        assert result.study_name == "test_study"
        assert result.n_trials == 5

    def test_reconstruction_is_deterministic(self):
        """Two FixedTrial reconstructions with the same params give identical cfg values."""
        params = _valid_default_params()
        cfg = _fresh_cfg()
        out1 = sample_space(_fixed_trial(params), cfg, DEFAULT_SPACE_SPEC)
        out2 = sample_space(_fixed_trial(params), cfg, DEFAULT_SPACE_SPEC)
        # They must agree on every sampled dimension
        assert out1.optim.learning_rate == pytest.approx(out2.optim.learning_rate)
        assert out1.sampler.ngh == out2.sampler.ngh
        assert out1.loss.cosim.weight == pytest.approx(out2.loss.cosim.weight)

    def test_reconstruction_with_partial_spec(self):
        """A partial spec (subset of DEFAULT_SPACE_SPEC) reconstructs correctly."""
        spec = {
            "optim.learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "sampler.ngh": {"type": "int", "low": 6, "high": 12},
        }
        params = {"optim.learning_rate": 5e-4, "sampler.ngh": 9}
        cfg = _fresh_cfg()
        best_cfg = sample_space(_fixed_trial(params), cfg, spec)
        assert best_cfg.optim.learning_rate == pytest.approx(5e-4)
        assert best_cfg.sampler.ngh == 9
        assert best_cfg.trainer.num_train_epochs == 3


# ---------------------------------------------------------------------------
# T-2.2: _assert_sampler_invariants standalone
# ---------------------------------------------------------------------------

class TestAssertSamplerInvariants:
    """_assert_sampler_invariants raises on bad combos, passes on good ones."""

    def _cfg_with(self, pos_d, neg_d, ngh):
        cfg = RunCfg()
        cfg.sampler.pos_d = pos_d
        cfg.sampler.neg_d = neg_d
        cfg.sampler.ngh = ngh
        return cfg

    def test_valid_baseline_passes(self):
        cfg = self._cfg_with(pos_d=3, neg_d=5, ngh=7)
        _assert_sampler_invariants(cfg)  # must not raise

    def test_pos_d_equals_neg_d_raises(self):
        cfg = self._cfg_with(pos_d=5, neg_d=5, ngh=7)
        with pytest.raises(ValueError, match="pos_d"):
            _assert_sampler_invariants(cfg)

    def test_neg_d_greater_than_ngh_raises(self):
        cfg = self._cfg_with(pos_d=2, neg_d=8, ngh=7)
        with pytest.raises(ValueError, match="ngh"):
            _assert_sampler_invariants(cfg)

    def test_tight_boundary_passes(self):
        """pos_d=1, neg_d=2, ngh=2 is the tightest valid combo."""
        cfg = self._cfg_with(pos_d=1, neg_d=2, ngh=2)
        _assert_sampler_invariants(cfg)  # must not raise


# ---------------------------------------------------------------------------
# T-2.3 (review fix): no-completed-trial sentinel
# ---------------------------------------------------------------------------

class TestRunStudySentinelContract:
    """run_study returns a sentinel StudyResult when zero trials complete.

    Sentinel contract (Phase 3 / T-3.4 detection):
        best_flow_MMA == float("-inf")   → no completed trial
        best_params   == {}
        best_cfg      is base_cfg        → unmodified base
    """

    def test_sentinel_when_all_trials_pruned(self):
        """If every trial raises TrialPruned, run_study returns sentinel (not crash)."""
        from unittest.mock import MagicMock, patch
        from optuna_search import run_study

        cfg = _fresh_cfg()
        # Use a minimal spec so sample_space doesn't raise itself.
        spec = {"optim.learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3}}

        # Make run_training return a trainer with best_metric=None so objective
        # raises TrialPruned (best_metric is None → TrialPruned, per objective logic).
        mock_trainer = MagicMock()
        mock_trainer.state.best_metric = None

        with patch("optuna_search.objective", side_effect=optuna.TrialPruned("no eval")):
            result = run_study(
                base_cfg=cfg,
                space_spec=spec,
                n_trials=2,
                study_name=f"test_sentinel_{id(cfg)}",
                storage="sqlite:///:memory:",
                datasets=MagicMock(),
            )

        assert result.best_flow_MMA == float("-inf"), (
            "Sentinel best_flow_MMA must be -inf when no trials completed"
        )
        assert result.best_params == {}, (
            "Sentinel best_params must be empty dict when no trials completed"
        )
        assert result.best_cfg is cfg, (
            "Sentinel best_cfg must be the unmodified base_cfg"
        )
        assert result.study_name is not None
        assert result.n_trials == 2

    def test_sentinel_detectable_by_caller(self):
        """Phase 3 can detect the sentinel via best_flow_MMA == -inf."""
        import math
        result = StudyResult(
            best_flow_MMA=float("-inf"),
            best_params={},
            best_cfg=RunCfg(),
            study_name="test",
            n_trials=3,
        )
        assert math.isinf(result.best_flow_MMA) and result.best_flow_MMA < 0
        assert result.best_params == {}


# ---------------------------------------------------------------------------
# T-2.2 (review fix): DEFAULT_SPACE_SPEC neg_d lower bound
# ---------------------------------------------------------------------------

class TestDefaultSpaceSpecNegDLowerBound:
    """neg_d lower bound must be >= 4 to prevent pos_d==neg_d==3 failed trials."""

    def test_neg_d_lower_bound_is_at_least_4(self):
        """Prevents pos_d (max 3) == neg_d combination from being sampled."""
        assert DEFAULT_SPACE_SPEC["sampler.neg_d"]["low"] >= 4, (
            "sampler.neg_d lower bound must be >= 4 so pos_d (max 3) < neg_d always holds"
        )

    def test_pos_d_upper_bound_less_than_neg_d_lower_bound(self):
        """pos_d upper bound must be strictly less than neg_d lower bound."""
        pos_d_high = DEFAULT_SPACE_SPEC["sampler.pos_d"]["high"]
        neg_d_low = DEFAULT_SPACE_SPEC["sampler.neg_d"]["low"]
        assert pos_d_high < neg_d_low, (
            f"pos_d max ({pos_d_high}) must be < neg_d min ({neg_d_low}) "
            "to guarantee pos_d < neg_d for all sampled combos"
        )
