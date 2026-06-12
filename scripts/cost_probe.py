"""Cost probe for a single 3-epoch Optuna trial.

Run this script on the training server to measure:
  - Wall-clock time for one 3-epoch trial
  - Peak VRAM consumed during that trial

Then use the printed calibration table to choose n_trials / timeout
for the Optuna study (optuna_search.py).

Usage (on training server, from repo root):
    ./env/bin/python scripts/cost_probe.py
    ./env/bin/python scripts/cost_probe.py --config config/default.yaml
    ./env/bin/python scripts/cost_probe.py --config config/default.yaml trainer.output_dir=/tmp/probe_out

The script builds datasets once (the slow HF Hub step), runs exactly one
training cycle with num_train_epochs=3, then prints wall-clock seconds,
peak VRAM in GB, and a back-of-envelope n_trials table for several
wall-clock budgets (2 h / 6 h / 12 h / 24 h).

NOTE — T-2.3 dependency:
    This script calls runner.run_training() directly because optuna_search.run_study()
    (T-2.3) was not yet complete when this script was written.  Once T-2.3 lands,
    the probe can optionally be updated to call run_study(..., n_trials=1) instead,
    which would also exercise the SQLite/TPESampler/pruner plumbing.
"""

import argparse
import sys
import time


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure wall-clock and peak-VRAM cost of one 3-epoch trial. "
            "Run on the training server; do not run locally (requires GPU + datasets)."
        )
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config (default: config/default.yaml)",
    )
    args, extra = parser.parse_known_args()
    return args, extra


def _print_calibration_table(elapsed_s: float) -> None:
    """Print a back-of-envelope n_trials table for common time budgets."""
    budgets = [
        ("2 h",  2 * 3600),
        ("6 h",  6 * 3600),
        ("12 h", 12 * 3600),
        ("24 h", 24 * 3600),
    ]
    col_w = 12
    print()
    print("Calibration table  (single-trial cost = {:.1f} s)".format(elapsed_s))
    print("-" * (col_w * 3 + 2))
    print("{:<{w}}  {:<{w}}  {:<{w}}".format("Budget", "Seconds", "n_trials", w=col_w))
    print("-" * (col_w * 3 + 2))
    for label, budget_s in budgets:
        n = int(budget_s / elapsed_s)
        print("{:<{w}}  {:<{w}}  {:<{w}}".format(label, str(budget_s), str(n), w=col_w))
    print("-" * (col_w * 3 + 2))
    print()
    print("Suggested defaults (PLACEHOLDER — replace after a real probe run):")
    print("  n_trials : use the '6 h' row as a starting point")
    print("  timeout  : set to budget_seconds * 0.95 to leave teardown slack")
    print()
    print("These numbers are per-GPU and do not account for pruned trials (which")
    print("are cheaper) or TPESampler warm-up (first ~10 trials are random).")


def run_probe(config_path: str, overrides: list) -> None:
    """Run one 3-epoch trial and report cost.

    This function contains all imports and heavy work so that --help and
    syntax checks work without triggering HF Hub downloads or GPU init.
    """
    import torch

    from config import load_config
    from build import build_datasets
    from runner import run_training

    # Force 3 epochs via an override so the probe always reflects the trial budget.
    all_overrides = list(overrides) + ["trainer.num_train_epochs=3"]
    cfg = load_config(config_path, *all_overrides)

    print("Building datasets (one-time cost, not included in trial timing)...")
    datasets = build_datasets(cfg)
    print("Datasets ready.")

    # Reset peak memory stats before the timed section.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("WARNING: CUDA not available — VRAM measurement will show 0.")

    t0 = time.perf_counter()
    trainer = run_training(cfg, datasets)
    elapsed = time.perf_counter() - t0

    peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_gb = peak_bytes / 1024 ** 3

    best_metric = getattr(trainer.state, "best_metric", None)

    print()
    print("=" * 50)
    print("Cost probe results")
    print("=" * 50)
    print(f"  Wall-clock time : {elapsed:.1f} s  ({elapsed / 60:.1f} min)")
    print(f"  Peak VRAM       : {peak_gb:.2f} GB")
    if best_metric is not None:
        print(f"  best flow_MMA   : {best_metric:.4f}")
    print("=" * 50)

    _print_calibration_table(elapsed)


if __name__ == "__main__":
    args, extra = _parse_args()
    run_probe(args.config, extra)
