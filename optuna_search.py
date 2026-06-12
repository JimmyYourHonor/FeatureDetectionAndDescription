"""Optuna search-space utilities for hyperparameter tuning.

Search-space spec format (contract shared with auto_research/search_space.py T-3.3):
    A dict mapping a RunCfg dotted-path to a distribution descriptor:

    {
        "optim.learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "sampler.ngh":         {"type": "int",   "low": 5,    "high": 15},
        "model.name":          {"type": "categorical", "choices": ["Quad_L2Net_ConfCFS"]},
    }

    Supported types:
      - "float"       -> trial.suggest_float(name, low, high, log=False)
                         Required keys: "low", "high"
                         Optional key: "log" (bool, default False) — use log-uniform sampling
      - "int"         -> trial.suggest_int(name, low, high)
                         Required keys: "low", "high"
      - "categorical" -> trial.suggest_categorical(name, choices)
                         Required key: "choices" (list)

    All dotted-path keys must be assignable into a RunCfg (validated by
    auto_research/search_space.py::validate_space_spec in T-3.3).

Excluded from the search space:
  - trainer.batch_eval_metrics  (setting False silently corrupts MMA — Phase-1 follow-up)
  - trainer.num_train_epochs    (forced to 3 by sample_space; not a search dimension)
"""

import argparse
import copy
import os
import uuid
from dataclasses import dataclass

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config import load_config
from config.schema import RunCfg


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------

DEFAULT_SPACE_SPEC: dict = {
    # --- optimiser ---
    "optim.learning_rate":  {"type": "float", "low": 5e-5, "high": 5e-3, "log": True},
    "optim.weight_decay":   {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "optim.warmup_steps":   {"type": "int",   "low": 100,  "high": 1000},

    # --- sampler ---
    # ngh controls the neighbourhood radius; pos_d/neg_d must satisfy pos_d < neg_d <= ngh.
    # Ranges keep pos_d in [1,3], neg_d in [4,6], ngh in [6,12] so the invariant
    # pos_d < neg_d <= ngh holds for all sampled combos. neg_d lower bound is 4
    # (not 3) to prevent the pos_d==neg_d==3 combination that causes FAILED trials.
    "sampler.ngh":          {"type": "int", "low": 6,  "high": 12},
    "sampler.pos_d":        {"type": "int", "low": 1,  "high": 3},
    "sampler.neg_d":        {"type": "int", "low": 4,  "high": 6},

    # --- loss weights ---
    "loss.reliability.weight": {"type": "float", "low": 0.5, "high": 2.0},
    "loss.cosim.weight":       {"type": "float", "low": 0.5, "high": 2.0},
    "loss.peaky.weight":       {"type": "float", "low": 0.5, "high": 2.0},

    # --- augmentation strength ---
    "augment.brightness":   {"type": "float", "low": 0.0, "high": 0.5},
    "augment.contrast":     {"type": "float", "low": 0.0, "high": 0.5},
    "augment.saturation":   {"type": "float", "low": 0.0, "high": 0.5},
    "augment.hue":          {"type": "float", "low": 0.0, "high": 0.2},
    "augment.noise_ampl":   {"type": "float", "low": 5.0, "high": 50.0},

    # --- eval NMS thresholds ---
    "eval.rel_thr":         {"type": "float", "low": 0.5, "high": 0.9},
    "eval.rep_thr":         {"type": "float", "low": 0.5, "high": 0.9},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_dotted(obj: object, dotted_path: str, value) -> None:
    """Set a value on a nested dataclass by dotted path.

    Example: _set_dotted(cfg, "optim.learning_rate", 1e-3)
    """
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _assert_sampler_invariants(cfg: RunCfg) -> None:
    """Re-assert NghSampler2 invariant: pos_d < neg_d <= ngh.

    Mirrors the check in build.py::build_sampler so invalid sampled
    combinations fail fast here rather than mid-epoch.
    """
    s = cfg.sampler
    if not (s.pos_d < s.neg_d <= s.ngh):
        raise ValueError(
            f"Sampler invariant violated after sampling: "
            f"pos_d ({s.pos_d}) < neg_d ({s.neg_d}) <= ngh ({s.ngh}) must hold. "
            f"Adjust the search space ranges so all combinations satisfy this."
        )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def sample_space(trial: optuna.Trial, base_cfg: RunCfg, space_spec: dict) -> RunCfg:
    """Deep-copy base_cfg and sample each path in space_spec from trial.

    After sampling, forces trainer.num_train_epochs = 3 and re-asserts
    the NghSampler2 invariant (pos_d < neg_d <= ngh).

    Args:
        trial:      An optuna Trial object.
        base_cfg:   The base RunCfg to copy and modify.
        space_spec: A search-space spec dict (see module docstring for format).

    Returns:
        A new RunCfg with sampled values and num_train_epochs forced to 3.

    Raises:
        ValueError: If the sampled combination violates sampler invariants.
        KeyError / TypeError: If a spec entry has an unsupported type or missing keys.
    """
    cfg = copy.deepcopy(base_cfg)

    for path, dist in space_spec.items():
        dist_type = dist["type"]
        if dist_type == "float":
            value = trial.suggest_float(
                path, dist["low"], dist["high"], log=dist.get("log", False)
            )
        elif dist_type == "int":
            value = trial.suggest_int(path, dist["low"], dist["high"])
        elif dist_type == "categorical":
            value = trial.suggest_categorical(path, dist["choices"])
        else:
            raise KeyError(
                f"Unsupported distribution type {dist_type!r} for path {path!r}. "
                f"Supported: 'float', 'int', 'categorical'."
            )
        _set_dotted(cfg, path, value)

    # Force the trial budget — never let a sampled spec override this.
    cfg.trainer.num_train_epochs = 3

    _assert_sampler_invariants(cfg)

    return cfg


# ---------------------------------------------------------------------------
# T-2.3 appends objective(trial, ...), run_study(...), and main() below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# StudyResult
# ---------------------------------------------------------------------------

@dataclass
class StudyResult:
    """Summary of a completed Optuna study."""
    best_flow_MMA: float
    best_params: dict
    best_cfg: RunCfg
    study_name: str
    n_trials: int


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, base_cfg: RunCfg, space_spec: dict, datasets) -> float:
    """Optuna objective for one trial.

    Samples a RunCfg from the search space, runs training, and returns
    trainer.state.best_metric (best eval_flow_MMA across epochs).

    TrialPruned propagates uncaught so Optuna records the prune.

    If best_metric is None (no successful evaluation completed — e.g. a very
    short run or a dataset issue), the trial is treated as pruned rather than
    recorded as a real score of 0.  Returning 0 would pollute the TPE model
    with a misleading "bad config = 0 MMA" signal.  Raising TrialPruned marks
    it as incomplete without penalising the sampled params.
    """
    from runner import run_training

    cfg = sample_space(trial, base_cfg, space_spec)
    trainer = run_training(cfg, datasets, trial=trial)
    value = trainer.state.best_metric

    if value is None:
        raise optuna.TrialPruned("best_metric is None — no eval completed")

    return value


# ---------------------------------------------------------------------------
# run_study
# ---------------------------------------------------------------------------

def run_study(
    base_cfg: RunCfg,
    space_spec: dict = None,
    n_trials: int = 20,
    study_name: str = None,
    storage: str = None,
    datasets=None,
) -> StudyResult:
    """Run an Optuna study over the given search space.

    Builds datasets once (slow HF Hub load) and reuses them across all
    in-process trials.  The study is persisted in SQLite and is resumable:
    calling run_study with the same study_name and storage resumes where it
    left off.

    Args:
        base_cfg:   Base RunCfg.  sample_space deep-copies it per trial.
        space_spec: Search-space spec dict.  Defaults to DEFAULT_SPACE_SPEC.
        n_trials:   Number of trials to run in this call.
        study_name: Optuna study name.  Generated from a UUID if not provided.
        storage:    Optuna storage URL.  Defaults to "sqlite:///optuna_studies.db".
        datasets:   Pre-built Datasets namedtuple.  Built once here if None.

    Returns:
        StudyResult with best_flow_MMA, best_params, best_cfg, study_name, n_trials.
    """
    from build import build_datasets

    if space_spec is None:
        space_spec = DEFAULT_SPACE_SPEC

    if study_name is None:
        study_name = f"hparam_search_{uuid.uuid4().hex[:8]}"

    if storage is None:
        storage = "sqlite:///optuna_studies.db"

    if datasets is None:
        datasets = build_datasets(base_cfg)

    # Route all trials in this study to a dedicated wandb group so runs are
    # grouped together in the W&B UI without breaking the runner's per-run
    # wandb setup.  We set WANDB_GROUP before optimize() and restore after.
    prev_group = os.environ.get("WANDB_GROUP")
    os.environ["WANDB_GROUP"] = study_name

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True),
        pruner=MedianPruner(n_warmup_steps=1),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    try:
        study.optimize(
            lambda t: objective(t, base_cfg, space_spec, datasets),
            n_trials=n_trials,
        )
    finally:
        if prev_group is None:
            os.environ.pop("WANDB_GROUP", None)
        else:
            os.environ["WANDB_GROUP"] = prev_group

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        # All trials were pruned or failed — no best trial available.
        # Return a sentinel so Phase 3 (T-3.4) can journal this as a discard
        # rather than receiving an unhandled ValueError crash.
        # Sentinel contract (detectable by callers):
        #   best_flow_MMA == float("-inf")  → no completed trial
        #   best_params   == {}
        #   best_cfg      == base_cfg (the unmodified base)
        return StudyResult(
            best_flow_MMA=float("-inf"),
            best_params={},
            best_cfg=base_cfg,
            study_name=study_name,
            n_trials=len(study.trials),
        )

    best_trial = study.best_trial
    best_cfg = sample_space(
        optuna.trial.FixedTrial(best_trial.params),
        base_cfg,
        space_spec,
    )

    return StudyResult(
        best_flow_MMA=best_trial.value,
        best_params=dict(best_trial.params),
        best_cfg=best_cfg,
        study_name=study_name,
        n_trials=len(study.trials),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(n_trials: int = 20) -> None:
    cfg = load_config("config/default.yaml")
    result = run_study(cfg, DEFAULT_SPACE_SPEC, n_trials)
    print(f"Study:          {result.study_name}")
    print(f"Trials run:     {result.n_trials}")
    print(f"Best flow MMA:  {result.best_flow_MMA:.4f}")
    print(f"Best params:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=20, dest="n_trials")
    parser.add_argument("--study-name", type=str, default=None, dest="study_name")
    parser.add_argument("--storage", type=str, default=None, dest="storage")
    args = parser.parse_args()

    cfg = load_config("config/default.yaml")
    result = run_study(
        cfg,
        DEFAULT_SPACE_SPEC,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
    )
    print(f"Study:          {result.study_name}")
    print(f"Trials run:     {result.n_trials}")
    print(f"Best flow MMA:  {result.best_flow_MMA:.4f}")
    print(f"Best params:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
