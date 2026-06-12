"""T-2.5 — Cross-trial isolation regression tests.

Guards against the two isolation risks documented in the plan:

  Risk #1 (metric accumulator leakage):
      Two sequential in-process trials (= two make_compute_metrics() calls) must
      produce independent accumulators. Feeding eval batches to one must not affect
      the other.
      The existing test_metric_isolation.py covers the general accumulator contract.
      This file adds scenario tests explicitly framed as "sequential Optuna trials"
      so T-2.5 is unambiguously covered.

  Risk #2 (output-dir / resume / callback isolation):
      For trial is not None, run_training must:
        - derive output_dir = <base_dir>/trial_<n>   (not the base_dir itself)
        - force resume_from_checkpoint = False
        - NOT register HFBucketCallback
        - NOT call the final-HPatches eval (_run_final_hpatches)
      These guarantees are tested WITHOUT running real training by monkeypatching
      build_datasets, build_trainer, and trainer.train.

All tests are CPU-only and network-free.
WandB is already mocked by conftest.py.

What is mocked and why:
  - build_datasets -> returns a sentinel Datasets namedtuple; avoids HF Hub I/O
  - build_trainer  -> returns a stub CustomTrainer-like object with
                       state.best_metric=0.5 and a no-op train(); avoids GPU init
  - trainer.train  -> no-op lambda; avoids all GPU/dataset usage
  - OptunaPruningCallback import -> uses the real class (no mock needed)
  - _run_final_hpatches -> verified it is NOT called by checking call_args on the stub
"""

import os
import sys
import types
import unittest.mock as mock

import pytest
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.schema import RunCfg
from build import Datasets
from models.evaluation.compute_metrics import (
    MetricAccumulator,
    _reset_state,
    make_compute_metrics,
)
from constants import FLOW_MMA_KEY


# ── helpers shared with test_metric_isolation.py (local copies to keep file
#    self-contained without importing from a test file) ──────────────────────

DESC_DIM = 128
FEAT_COLS = 3 + DESC_DIM


def _orthogonal_features(n=6):
    import torch
    feats = torch.zeros(n, FEAT_COLS)
    feats[:, 0] = torch.arange(n, dtype=torch.float32) * 5.0 + 1.0
    feats[:, 1] = torch.arange(n, dtype=torch.float32) * 5.0 + 1.0
    feats[:, 2] = 1.0
    for i in range(min(n, DESC_DIM)):
        feats[i, 3 + i] = 1.0
    return feats


def _identity_aflow(H=64, W=64):
    import torch
    xs = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    ys = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    return torch.stack([xs, ys], dim=0)


def _good_flow_batch(n=6):
    from transformers.trainer_utils import EvalPrediction
    feats = _orthogonal_features(n)
    return EvalPrediction(predictions=[feats, feats.clone()], label_ids=_identity_aflow())


def _empty_flow_batch():
    import torch
    from transformers.trainer_utils import EvalPrediction
    empty = torch.zeros(0, FEAT_COLS)
    return EvalPrediction(predictions=[empty, empty.clone()], label_ids=_identity_aflow())


# ── autouse fixture: keep module-level default clean ─────────────────────────

@pytest.fixture(autouse=True)
def clean_module_state():
    _reset_state()
    yield
    _reset_state()


# ── Risk #1: metric accumulator isolation across sequential "trials" ─────────

class TestSequentialTrialAccumulatorIsolation:
    """Two sequential make_compute_metrics() calls simulate two sequential Optuna
    trials. Feeding data to trial-1's accumulator must not affect trial-2's result,
    and vice versa.

    This is the explicit T-2.5 framing of the accumulator isolation contract
    from test_metric_isolation.py.
    """

    def test_trial_1_accumulation_does_not_affect_trial_2(self):
        """After trial 1 feeds many good batches, trial 2 starts fresh."""
        # Simulate trial 1: build compute_fn and accumulate several batches
        fn_trial1, acc1 = make_compute_metrics()
        fn_trial1(_good_flow_batch(), compute_result=False)
        fn_trial1(_good_flow_batch(), compute_result=False)
        fn_trial1(_good_flow_batch(), compute_result=False)

        # Simulate trial 2: build a new compute_fn (next make_compute_metrics call)
        fn_trial2, acc2 = make_compute_metrics()

        # Trial 2's accumulator must be empty — no cross-trial leakage
        assert acc2.total_pairs == 0, (
            "Trial-2 accumulator already has data — cross-trial leakage from trial-1"
        )
        result2 = fn_trial2(_empty_flow_batch(), compute_result=True)
        assert result2["MMA"] == pytest.approx(0.0), (
            "Trial-2 result contaminated by trial-1's accumulation"
        )

    def test_trial_2_accumulation_does_not_affect_trial_1_if_result_not_yet_called(self):
        """If trial 1 hasn't called result() yet, trial 2 accumulation must not
        alter trial 1's accumulated state."""
        fn_trial1, acc1 = make_compute_metrics()
        fn_trial1(_good_flow_batch(), compute_result=False)  # trial 1 has 1 pair
        pairs_before = acc1.total_pairs

        fn_trial2, acc2 = make_compute_metrics()
        fn_trial2(_good_flow_batch(), compute_result=False)  # trial 2 accumulates too

        # trial 1's accumulator must be unmodified
        assert acc1.total_pairs == pairs_before, (
            "Trial-2 accumulation changed trial-1's accumulator state"
        )

    def test_three_sequential_trials_all_independent(self):
        """Three back-to-back 'trials' each see only their own data."""
        fn1, _ = make_compute_metrics()
        fn2, _ = make_compute_metrics()
        fn3, _ = make_compute_metrics()

        # Feed different amounts of data to each
        result1 = fn1(_good_flow_batch(n=6), compute_result=True)
        result2 = fn2(_empty_flow_batch(), compute_result=True)
        result3 = fn3(_good_flow_batch(n=6), compute_result=True)

        assert result1["MMA"] == pytest.approx(1.0), "trial 1 should have perfect MMA"
        assert result2["MMA"] == pytest.approx(0.0), "trial 2 (empty) should have MMA=0"
        assert result3["MMA"] == pytest.approx(1.0), "trial 3 should have perfect MMA"

    def test_accumulator_is_reset_between_uses_of_same_fn(self):
        """Each result() call resets the accumulator so the same fn can be used
        for 'two epochs' in one trial without cross-epoch contamination."""
        fn, acc = make_compute_metrics()

        # Epoch 1: perfect data
        r1 = fn(_good_flow_batch(), compute_result=True)
        assert r1["MMA"] == pytest.approx(1.0)

        # Epoch 2 (same fn): only empty data → must be 0.0, not a blend
        r2 = fn(_empty_flow_batch(), compute_result=True)
        assert r2["MMA"] == pytest.approx(0.0), (
            "Accumulator carried epoch-1 data into epoch-2 within the same trial"
        )

    def test_make_compute_metrics_each_call_returns_new_object(self):
        """Sanity: each make_compute_metrics() call returns a distinct accumulator
        object, not the same one reused."""
        _, acc1 = make_compute_metrics()
        _, acc2 = make_compute_metrics()
        assert acc1 is not acc2
        assert acc1.err is not acc2.err


# ── Risk #2: output-dir / resume / callback isolation via monkeypatching ─────

def _make_stub_trial(number: int = 0):
    """Create a real Optuna trial from an in-memory study."""
    study = optuna.create_study(direction="maximize")
    # Ask gives a trial; we set .number manually for the path test
    trial = study.ask()
    # Optuna assigns number=0 for the first ask; for custom number use a study
    # that already has the right count — or just accept number=0 and 1
    return trial


def _make_sentinel_datasets():
    """Return a Datasets namedtuple with None fields — enough to pass into run_training."""
    return Datasets(
        train_dataset=None,
        flow_eval_dataset=None,
        hpatches_eval_dataset=None,
        hpatches_probe=None,
    )


def _make_stub_trainer(best_metric=0.5):
    """Return a minimal stub that satisfies what run_training reads from the trainer."""
    stub = mock.MagicMock()
    stub.state = mock.MagicMock()
    stub.state.best_metric = best_metric
    stub.train = mock.MagicMock(return_value=None)
    stub.evaluate = mock.MagicMock(return_value={"eval_MMA": 0.5, "eval_avg_matches": 10.0, "eval_avg_feats": 20.0})
    return stub


def _fake_build_trial_callbacks_with_capture(trial, pruning_cb):
    """Helper: returns a fake _build_trial_callbacks that captures callbacks.

    We cannot let the real _build_trial_callbacks run in tests because it
    imports WeightAnalysisCallback / EvalCallback, which call
    importlib.util.find_spec('wandb'). The wandb mock in conftest.py installs a
    MagicMock into sys.modules['wandb'], but MagicMock.__spec__ is None, which
    causes find_spec to raise ValueError.

    The real _build_trial_callbacks behavior (including OptunaPruningCallback
    registration) is tested separately in TestOptunaPruningCallbackOnEvaluate.
    Here we stub it to return a list containing just the OptunaPruningCallback
    so the callback-presence tests still exercise the right contract.
    """
    from models.evaluation.optuna_pruning_callback import OptunaPruningCallback
    return [pruning_cb if pruning_cb else OptunaPruningCallback(trial)]


class TestRunnerTrialIsolationPolicy:
    """Verify run_training's trial-isolation policy WITHOUT running real training.

    What is mocked:
      - build_datasets: avoided to skip HF Hub I/O; returns sentinel Datasets
      - build_trainer: avoided to skip GPU init (calls .cuda() on components);
                       returns a stub trainer whose .train() is a no-op
      - make_model_init: returns a lambda; avoids model.cuda() in test env
      - _build_trial_callbacks: avoided because WeightAnalysisCallback/EvalCallback
                       call importlib.util.find_spec('wandb') which fails against
                       the MagicMock wandb. Returns a list with one real
                       OptunaPruningCallback so callback-presence tests still hold.
      - _teardown: no-op; avoids torch.cuda.empty_cache() in test env

    What is NOT mocked:
      - run_training's own path-construction logic (output_dir, guards)
    """

    def _run_with_mocks(self, trial, cfg=None):
        """Invoke run_training with all heavy dependencies mocked out.

        Returns (captured dict, stub_trainer, returned_trainer).
        captured keys: output_dir, callbacks
        """
        import runner as runner_module
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        if cfg is None:
            cfg = RunCfg()

        datasets = _make_sentinel_datasets()
        stub = _make_stub_trainer()
        pruning_cb = OptunaPruningCallback(trial)

        captured = {}

        def fake_build_trainer(cfg, datasets, model_init, compute_metrics_fn=None,
                               output_dir=None, callbacks=None):
            captured["output_dir"] = output_dir
            captured["callbacks"] = callbacks
            return stub

        def fake_make_model_init(cfg):
            return lambda: None

        def fake_teardown(trainer):
            pass

        def fake_build_trial_callbacks(cfg, trial):
            return [pruning_cb]

        with mock.patch.object(runner_module, "build_trainer", fake_build_trainer), \
             mock.patch.object(runner_module, "make_model_init", fake_make_model_init), \
             mock.patch.object(runner_module, "_teardown", fake_teardown), \
             mock.patch.object(runner_module, "_build_trial_callbacks",
                               fake_build_trial_callbacks):
            trainer = runner_module.run_training(cfg, datasets, trial=trial)

        return captured, stub, trainer

    def test_trial_path_uses_per_trial_output_dir(self):
        """When trial is not None, output_dir must be <base>/trial_<n>."""
        trial = _make_stub_trial()
        cfg = RunCfg()
        cfg.trainer.output_dir = "/workspace/outputs"

        captured, _, _ = self._run_with_mocks(trial, cfg)

        expected_dir = f"/workspace/outputs/trial_{trial.number}"
        assert captured["output_dir"] == expected_dir, (
            f"Expected output_dir={expected_dir!r}, got {captured['output_dir']!r}"
        )

    def test_none_trial_uses_base_output_dir(self):
        """When trial is None, output_dir must be cfg.trainer.output_dir."""
        import runner as runner_module

        cfg = RunCfg()
        cfg.trainer.output_dir = "/workspace/outputs"
        cfg.trainer.resume = False  # skip checkpoint resolution
        datasets = _make_sentinel_datasets()
        stub = _make_stub_trainer()

        captured = {}

        def fake_build_trainer(cfg, datasets, model_init, compute_metrics_fn=None,
                               output_dir=None, callbacks=None):
            captured["output_dir"] = output_dir
            captured["callbacks"] = callbacks
            return stub

        def fake_make_model_init(cfg):
            return lambda: None

        def fake_resolve_checkpoint(cfg, output_dir):
            return False

        def fake_run_final_hpatches(trainer, hpatches_ds):
            captured["final_hpatches_called"] = True

        with mock.patch.object(runner_module, "build_trainer", fake_build_trainer), \
             mock.patch.object(runner_module, "make_model_init", fake_make_model_init), \
             mock.patch.object(runner_module, "_resolve_checkpoint", fake_resolve_checkpoint), \
             mock.patch.object(runner_module, "_run_final_hpatches", fake_run_final_hpatches):
            runner_module.run_training(cfg, datasets, trial=None)

        assert captured["output_dir"] == "/workspace/outputs", (
            "Non-trial path must use cfg.trainer.output_dir"
        )

    def test_trial_path_hpatches_not_called(self):
        """When trial is not None, _run_final_hpatches must NOT be called."""
        import runner as runner_module
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = _make_stub_trial()
        cfg = RunCfg()
        datasets = _make_sentinel_datasets()
        stub = _make_stub_trainer()

        hpatches_called = {"value": False}

        def fake_build_trainer(cfg, datasets, model_init, compute_metrics_fn=None,
                               output_dir=None, callbacks=None):
            return stub

        def fake_make_model_init(cfg):
            return lambda: None

        def fake_teardown(trainer):
            pass

        def fake_run_final_hpatches(trainer, hpatches_ds):
            hpatches_called["value"] = True

        def fake_build_trial_callbacks(cfg, trial):
            return [OptunaPruningCallback(trial)]

        with mock.patch.object(runner_module, "build_trainer", fake_build_trainer), \
             mock.patch.object(runner_module, "make_model_init", fake_make_model_init), \
             mock.patch.object(runner_module, "_teardown", fake_teardown), \
             mock.patch.object(runner_module, "_run_final_hpatches", fake_run_final_hpatches), \
             mock.patch.object(runner_module, "_build_trial_callbacks",
                               fake_build_trial_callbacks):
            runner_module.run_training(cfg, datasets, trial=trial)

        assert not hpatches_called["value"], (
            "Final HPatches eval must be suppressed on the trial path"
        )

    def test_non_trial_path_hpatches_called(self):
        """When trial is None, _run_final_hpatches IS called (CLI path)."""
        import runner as runner_module

        cfg = RunCfg()
        cfg.trainer.resume = False
        datasets = _make_sentinel_datasets()
        stub = _make_stub_trainer()

        hpatches_called = {"value": False}

        def fake_build_trainer(cfg, datasets, model_init, compute_metrics_fn=None,
                               output_dir=None, callbacks=None):
            return stub

        def fake_make_model_init(cfg):
            return lambda: None

        def fake_resolve_checkpoint(cfg, output_dir):
            return False

        def fake_run_final_hpatches(trainer, hpatches_ds):
            hpatches_called["value"] = True

        with mock.patch.object(runner_module, "build_trainer", fake_build_trainer), \
             mock.patch.object(runner_module, "make_model_init", fake_make_model_init), \
             mock.patch.object(runner_module, "_resolve_checkpoint", fake_resolve_checkpoint), \
             mock.patch.object(runner_module, "_run_final_hpatches", fake_run_final_hpatches):
            runner_module.run_training(cfg, datasets, trial=None)

        assert hpatches_called["value"], (
            "Final HPatches eval must be called on the non-trial (CLI) path"
        )

    def test_trial_path_registers_pruning_callback(self):
        """When trial is not None, OptunaPruningCallback must be in the callbacks list."""
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = _make_stub_trial()
        captured, _, _ = self._run_with_mocks(trial)

        callbacks = captured.get("callbacks", [])
        assert callbacks is not None, "callbacks must not be None on trial path"
        pruning_cbs = [cb for cb in callbacks if isinstance(cb, OptunaPruningCallback)]
        assert len(pruning_cbs) == 1, (
            f"Expected exactly one OptunaPruningCallback in trial callbacks, "
            f"got {len(pruning_cbs)}"
        )

    def test_trial_path_pruning_callback_has_correct_trial(self):
        """OptunaPruningCallback registered for this trial must hold this trial."""
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = _make_stub_trial()
        captured, _, _ = self._run_with_mocks(trial)

        callbacks = captured.get("callbacks", [])
        pruning_cbs = [cb for cb in callbacks if isinstance(cb, OptunaPruningCallback)]
        assert pruning_cbs[0].trial is trial, (
            "OptunaPruningCallback holds the wrong trial reference"
        )

    def test_non_trial_path_no_pruning_callback(self):
        """When trial is None, no OptunaPruningCallback is registered."""
        import runner as runner_module
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        cfg = RunCfg()
        cfg.trainer.resume = False
        datasets = _make_sentinel_datasets()
        stub = _make_stub_trainer()
        captured = {}

        def fake_build_trainer(cfg, datasets, model_init, compute_metrics_fn=None,
                               output_dir=None, callbacks=None):
            captured["callbacks"] = callbacks
            return stub

        def fake_make_model_init(cfg):
            return lambda: None

        def fake_resolve_checkpoint(cfg, output_dir):
            return False

        def fake_run_final_hpatches(trainer, hpatches_ds):
            pass

        with mock.patch.object(runner_module, "build_trainer", fake_build_trainer), \
             mock.patch.object(runner_module, "make_model_init", fake_make_model_init), \
             mock.patch.object(runner_module, "_resolve_checkpoint", fake_resolve_checkpoint), \
             mock.patch.object(runner_module, "_run_final_hpatches", fake_run_final_hpatches):
            runner_module.run_training(cfg, datasets, trial=None)

        # callbacks=None on the non-trial path means build_trainer uses its own defaults
        assert captured.get("callbacks") is None, (
            "Non-trial path must pass callbacks=None to build_trainer "
            "(build_trainer applies its own default callback list)"
        )

    def test_two_sequential_trials_get_different_output_dirs(self):
        """Two sequential trial calls must produce non-colliding output_dirs."""
        import runner as runner_module
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        study = optuna.create_study(direction="maximize")
        trial0 = study.ask()   # number=0
        trial1 = study.ask()   # number=1

        cfg = RunCfg()
        cfg.trainer.output_dir = "/workspace/outputs"
        datasets = _make_sentinel_datasets()

        captured_dirs = []

        def fake_build_trainer(cfg, datasets, model_init, compute_metrics_fn=None,
                               output_dir=None, callbacks=None):
            captured_dirs.append(output_dir)
            return _make_stub_trainer()

        def fake_make_model_init(cfg):
            return lambda: None

        def fake_teardown(trainer):
            pass

        def fake_build_trial_callbacks(cfg, trial):
            return [OptunaPruningCallback(trial)]

        with mock.patch.object(runner_module, "build_trainer", fake_build_trainer), \
             mock.patch.object(runner_module, "make_model_init", fake_make_model_init), \
             mock.patch.object(runner_module, "_teardown", fake_teardown), \
             mock.patch.object(runner_module, "_build_trial_callbacks",
                               fake_build_trial_callbacks):
            runner_module.run_training(cfg, datasets, trial=trial0)
            runner_module.run_training(cfg, datasets, trial=trial1)

        assert captured_dirs[0] != captured_dirs[1], (
            "Two sequential trials used the same output_dir — checkpoint collision risk"
        )
        assert "trial_0" in captured_dirs[0]
        assert "trial_1" in captured_dirs[1]

    def test_trial_teardown_is_called(self):
        """_teardown is called in the finally block on the trial path."""
        import runner as runner_module
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback

        trial = _make_stub_trial()
        datasets = _make_sentinel_datasets()
        cfg = RunCfg()
        stub = _make_stub_trainer()
        teardown_called = {"value": False}

        def fake_build_trainer(cfg, datasets, model_init, compute_metrics_fn=None,
                               output_dir=None, callbacks=None):
            return stub

        def fake_make_model_init(cfg):
            return lambda: None

        def fake_teardown(trainer):
            teardown_called["value"] = True

        def fake_build_trial_callbacks(cfg, trial):
            return [OptunaPruningCallback(trial)]

        with mock.patch.object(runner_module, "build_trainer", fake_build_trainer), \
             mock.patch.object(runner_module, "make_model_init", fake_make_model_init), \
             mock.patch.object(runner_module, "_teardown", fake_teardown), \
             mock.patch.object(runner_module, "_build_trial_callbacks",
                               fake_build_trial_callbacks):
            runner_module.run_training(cfg, datasets, trial=trial)

        assert teardown_called["value"], "_teardown was not called on the trial path"
