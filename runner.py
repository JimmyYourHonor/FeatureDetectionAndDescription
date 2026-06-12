import os
from safetensors.torch import load_file

from config.schema import RunCfg
from constants import HPATCHES_MMA_KEY
from models.evaluation.compute_metrics import make_compute_metrics
from build import (
    Datasets,
    build_datasets,
    build_model,
    build_sampler,
    build_loss,
    build_trainer,
    make_model_init,
)


def run_training(cfg: RunCfg, datasets: Datasets = None, *, trial=None):
    """Run one full training cycle and return the trainer.

    Args:
        cfg:      Full RunCfg. For trials, caller should already have set
                  num_train_epochs=3 (or similar) via sample_space.
        datasets: Pre-built Datasets namedtuple. When None, build_datasets(cfg)
                  is called (slow HF Hub loads). Pass the same object across
                  trials to avoid re-downloading.
        trial:    An optuna.Trial when called from the Optuna objective, or None
                  for the CLI path. Controls trial-isolation policy (see below).

    Returns:
        The trained CustomTrainer. Caller can read trainer.state.best_metric.

    Trial-isolation policy (trial is not None):
        - output_dir is set to <base>/trial_<n> to avoid checkpoint collisions
        - resume is forced off (no checkpoint loading)
        - WANDB_RUN_ID injection is suppressed (no resume of prior wandb run)
        - HFBucketCallback S3 upload is disabled
        - Final HPatches evaluation is skipped
        - OptunaPruningCallback is registered for per-epoch pruning
    """
    if datasets is None:
        datasets = build_datasets(cfg)

    compute_fn, _ = make_compute_metrics()
    model_init = make_model_init(cfg)

    is_trial = trial is not None

    if is_trial:
        base_dir = cfg.trainer.output_dir
        output_dir = os.path.join(base_dir, f"trial_{trial.number}")
        resume_from_checkpoint = False
        callbacks = _build_trial_callbacks(cfg, trial)
    else:
        output_dir = cfg.trainer.output_dir
        resume_from_checkpoint = _resolve_checkpoint(cfg, output_dir)
        callbacks = None  # build_trainer uses the default callbacks

    trainer = build_trainer(
        cfg,
        datasets,
        model_init,
        compute_metrics_fn=compute_fn,
        output_dir=output_dir,
        callbacks=callbacks,
    )

    if not is_trial and resume_from_checkpoint:
        _inject_wandb_run_id(resume_from_checkpoint)

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        if not is_trial:
            _run_final_hpatches(trainer, datasets.hpatches_eval_dataset)
    finally:
        if is_trial:
            _teardown(trainer)

    return trainer


def _build_trial_callbacks(cfg, trial):
    from callbacks.weight_analysis_callback import WeightAnalysisCallback
    from models.evaluation.eval_callback import EvalCallback

    cbs = [WeightAnalysisCallback(), EvalCallback()]

    try:
        from models.evaluation.optuna_pruning_callback import OptunaPruningCallback
        cbs.append(OptunaPruningCallback(trial))
    except ImportError:
        pass

    return cbs


def _resolve_checkpoint(cfg, output_dir):
    from callbacks.hf_bucket_callback import HFBucketCallback

    if not cfg.trainer.resume:
        return False
    if os.path.isdir(output_dir) and os.path.isfile(
        os.path.join(output_dir, "checkpoint-latest")
    ):
        return output_dir + "/checkpoint-latest"
    local_checkpoint = HFBucketCallback.download_latest_checkpoint(
        output_dir, cfg.name
    )
    if local_checkpoint:
        return local_checkpoint
    return False


def _inject_wandb_run_id(resume_from_checkpoint):
    ckpt_path = (
        resume_from_checkpoint
        if isinstance(resume_from_checkpoint, str)
        else None
    )
    id_file = os.path.join(ckpt_path, "wandb_run_id.txt") if ckpt_path else None
    if id_file and os.path.isfile(id_file):
        with open(id_file) as f:
            os.environ["WANDB_RUN_ID"] = f.read().strip()
        os.environ["WANDB_RESUME"] = "allow"


def _run_final_hpatches(trainer, hpatches_eval_dataset):
    print("Running final evaluation on HPatches sequences...")
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint:
        print(f"Loading best model checkpoint from {best_checkpoint} for HPatches evaluation...")
        trainer.model.load_state_dict(
            load_file(os.path.join(best_checkpoint, "model.safetensors"))
        )
    hpatches_metrics = trainer.evaluate(eval_dataset=hpatches_eval_dataset)
    print(f"HPatches MMA:         {hpatches_metrics[HPATCHES_MMA_KEY]:.4f}")
    print(f"HPatches avg matches: {hpatches_metrics['eval_avg_matches']:.1f}")
    print(f"HPatches avg feats:   {hpatches_metrics['eval_avg_feats']:.1f}")


def _teardown(trainer):
    import torch
    # Delete all CUDA-resident modules and optimizer state attached to the
    # trainer. trainer itself is returned to the caller (for state.best_metric
    # access), so we cannot delete trainer — only its CUDA-heavy attributes.
    for attr in ("model", "loss", "augment", "warp", "window_select", "optimizer", "lr_scheduler"):
        try:
            delattr(trainer, attr)
        except AttributeError:
            pass
    torch.cuda.empty_cache()
