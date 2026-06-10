import os
import warnings
from typing import Callable, NamedTuple

import datasets as hf_datasets
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import TrainingArguments

from config.schema import RunCfg
from constants import FLOW_MMA_UNPREFIXED
from models import (
    ConvNeXtV2,
    CosimLoss,
    CustomTrainer,
    MultiLoss,
    NghSampler2,
    PeakyLoss,
    Quad_L2Net_ConfCFS,
    ReliabilityLoss,
    ViTDense,
)
from models.evaluation.compute_metrics import make_compute_metrics
from preprocessing import (
    GPUBatchAugment,
    GPUWarp,
    GPUWindowSelect,
    ParametricTransform,
    parametric_collator,
)
from preprocessing.transform_builder import RandomScale, RandomTilting
from callbacks.weight_analysis_callback import WeightAnalysisCallback
from callbacks.hf_bucket_callback import HFBucketCallback
from models.evaluation.eval_callback import EvalCallback


class Datasets(NamedTuple):
    train_dataset: hf_datasets.Dataset
    flow_eval_dataset: hf_datasets.Dataset
    hpatches_eval_dataset: hf_datasets.Dataset
    hpatches_probe: hf_datasets.Dataset


def build_model(cfg: RunCfg) -> nn.Module:
    name = cfg.model.name
    if name == "Quad_L2Net_ConfCFS":
        return Quad_L2Net_ConfCFS(
            dim=cfg.model.dim,
            mchan=cfg.model.mchan,
            relu22=cfg.model.relu22,
        )
    elif name == "ConvnextV2":
        return ConvNeXtV2(model_scale=cfg.model.convnext_scale)
    elif name == "ViTDense":
        return ViTDense()
    else:
        raise ValueError(f"Unknown model name: {name!r}")


def make_model_init(cfg: RunCfg) -> Callable[[], nn.Module]:
    def model_init():
        return build_model(cfg)
    return model_init


def build_sampler(cfg: RunCfg) -> NghSampler2:
    s = cfg.sampler
    if not (s.pos_d < s.neg_d <= s.ngh):
        raise ValueError(
            f"Sampler invariant violated: pos_d ({s.pos_d}) < neg_d ({s.neg_d})"
            f" <= ngh ({s.ngh}) must hold."
        )
    return NghSampler2(
        ngh=s.ngh,
        subq=s.subq,
        subd=s.subd,
        pos_d=s.pos_d,
        neg_d=s.neg_d,
        border=s.border,
        subd_neg=s.subd_neg,
        maxpool_pos=s.maxpool_pos,
    )


def build_loss(cfg: RunCfg, sampler: NghSampler2) -> MultiLoss:
    rel_p = cfg.loss.reliability.params
    cos_p = cfg.loss.cosim.params
    pky_p = cfg.loss.peaky.params
    return MultiLoss(
        cfg.loss.reliability.weight,
        ReliabilityLoss(sampler, base=rel_p["base"], nq=rel_p["nq"]),
        cfg.loss.cosim.weight,
        CosimLoss(N=cos_p["N"]),
        cfg.loss.peaky.weight,
        PeakyLoss(N=pky_p["N"]),
    )


def build_training_args(cfg: RunCfg, output_dir_override: str = None) -> TrainingArguments:
    from transformers.training_args import OptimizerNames
    from transformers.trainer_utils import IntervalStrategy

    # Resolve optimizer: schedule_free_adamw is only in transformers >=4.46.
    optim_values = {e.value for e in OptimizerNames}
    if cfg.optim.optim in optim_values:
        optim = cfg.optim.optim
    else:
        warnings.warn(
            f"Optimizer {cfg.optim.optim!r} not available in this transformers version; "
            "falling back to 'adamw_torch'. Upgrade to transformers >=4.46 for the correct optimizer.",
            RuntimeWarning,
            stacklevel=2,
        )
        optim = "adamw_torch"

    # Resolve save_strategy: "best" is only in transformers >=4.46.
    interval_values = {e.value for e in IntervalStrategy}
    if cfg.trainer.save_strategy in interval_values:
        save_strategy = cfg.trainer.save_strategy
    else:
        warnings.warn(
            f"save_strategy {cfg.trainer.save_strategy!r} not available in this transformers version; "
            "falling back to 'epoch'. Upgrade to transformers >=4.46 for the correct save strategy.",
            RuntimeWarning,
            stacklevel=2,
        )
        save_strategy = "epoch"

    # evaluation_strategy is the <=4.45 name; eval_strategy was added in 4.46.
    # We use a try/except on the keyword name to stay forward-compatible.
    eval_strat_key = "eval_strategy" if _has_arg(TrainingArguments, "eval_strategy") else "evaluation_strategy"

    kwargs = dict(
        output_dir=output_dir_override if output_dir_override is not None else cfg.trainer.output_dir,
        optim=optim,
        lr_scheduler_type=cfg.optim.lr_scheduler_type,
        learning_rate=cfg.optim.learning_rate,
        warmup_steps=cfg.optim.warmup_steps,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        num_train_epochs=cfg.trainer.num_train_epochs,
        weight_decay=cfg.optim.weight_decay,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        save_strategy=save_strategy,
        logging_steps=cfg.trainer.logging_steps,
        remove_unused_columns=False,
        report_to=cfg.trainer.report_to,
        metric_for_best_model=FLOW_MMA_UNPREFIXED,
        greater_is_better=True,
        max_grad_norm=cfg.optim.max_grad_norm,
        bf16=cfg.trainer.bf16,
        seed=cfg.trainer.seed,
    )
    kwargs[eval_strat_key] = cfg.trainer.eval_strategy

    if _has_arg(TrainingArguments, "batch_eval_metrics"):
        kwargs["batch_eval_metrics"] = cfg.trainer.batch_eval_metrics
    else:
        warnings.warn(
            "batch_eval_metrics not available in this transformers version; "
            "MetricAccumulator per-batch contract will not hold. "
            "Upgrade to transformers >=4.46 for correct MMA computation.",
            RuntimeWarning,
            stacklevel=2,
        )

    return TrainingArguments(**kwargs)


def _has_arg(cls, name: str) -> bool:
    import inspect
    return name in inspect.signature(cls.__init__).parameters


def build_datasets(cfg: RunCfg) -> Datasets:
    web_images_train = hf_datasets.load_dataset("JimmyFu/web_images", split="train[:90%]")
    web_images_val = hf_datasets.load_dataset("JimmyFu/web_images", split="train[90%:]")
    aachen_db_train = hf_datasets.load_dataset("JimmyFu/aachen_db_images", split="train[:90%]")
    aachen_db_val = hf_datasets.load_dataset("JimmyFu/aachen_db_images", split="train[90%:]")
    aachen_st_train = hf_datasets.load_dataset("JimmyFu/aachen_style_transfer", split="train[:90%]")
    aachen_st_val = hf_datasets.load_dataset("JimmyFu/aachen_style_transfer", split="train[90%:]")
    aachen_fp_train = hf_datasets.load_dataset("JimmyFu/aachen_flow_pairs", split="train[:90%]")
    aachen_fp_val = hf_datasets.load_dataset("JimmyFu/aachen_flow_pairs", split="train[90%:]")

    train_dataset = hf_datasets.interleave_datasets([
        web_images_train, aachen_db_train, aachen_st_train, aachen_fp_train
    ])

    flow_eval_dataset = hf_datasets.interleave_datasets([
        web_images_val, aachen_db_val, aachen_st_val, aachen_fp_val
    ])
    n_eval = min(1000, len(flow_eval_dataset))
    flow_eval_dataset = flow_eval_dataset.select(range(n_eval))

    hpatches_eval_dataset = hf_datasets.load_dataset("JimmyFu/hpatches_sequences", split="eval")

    image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def apply_transforms(examples):
        output_examples = {}
        for i in range(1, 7):
            output_examples[f"{i}.ppm"] = [image_transforms(img) for img in examples[f"{i}.ppm"]]
            if i != 1:
                output_examples[f"h_1_{i}"] = [
                    torch.from_numpy(np.array(h)) for h in examples[f"h_1_{i}"]
                ]
        return output_examples

    hpatches_eval_dataset.set_transform(apply_transforms)

    n_probe = min(150, len(hpatches_eval_dataset))
    hpatches_probe = hpatches_eval_dataset.select(range(n_probe))
    hpatches_probe.set_transform(apply_transforms)

    transform = ParametricTransform(
        synthetic_scale=RandomScale(cfg.transform.scale_min, cfg.transform.scale_max, can_upscale=cfg.transform.scale_can_upscale),
        synthetic_tilt=RandomTilting(cfg.transform.tilt_magnitude),
        still_scale=RandomScale(cfg.transform.scale_min, cfg.transform.scale_max, can_upscale=cfg.transform.scale_can_upscale),
        still_tilt=RandomTilting(cfg.transform.tilt_magnitude),
        second_scale=RandomScale(cfg.transform.scale_min, cfg.transform.scale_max, can_upscale=cfg.transform.scale_can_upscale),
    )
    train_dataset.set_transform(transform)
    flow_eval_dataset.set_transform(transform)

    return Datasets(
        train_dataset=train_dataset,
        flow_eval_dataset=flow_eval_dataset,
        hpatches_eval_dataset=hpatches_eval_dataset,
        hpatches_probe=hpatches_probe,
    )


def build_trainer(
    cfg: RunCfg,
    datasets: Datasets,
    model_init: Callable[[], nn.Module],
    compute_metrics_fn=None,
    output_dir: str = None,
    callbacks=None,
) -> CustomTrainer:
    training_args = build_training_args(cfg, output_dir_override=output_dir)

    if compute_metrics_fn is None:
        compute_metrics_fn, _ = make_compute_metrics()

    if callbacks is None:
        model_name = cfg.model.name
        callbacks = [
            WeightAnalysisCallback(),
            EvalCallback(),
            HFBucketCallback(model_name),
        ]

    trainer = CustomTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset={"flow": datasets.flow_eval_dataset, "hpatches": datasets.hpatches_probe},
        data_collator=parametric_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics_fn,
        eval_cfg=cfg.eval,
    )

    sampler = build_sampler(cfg)
    loss = build_loss(cfg, sampler)

    augment_cfg = cfg.augment
    trainer.set_loss(loss.cuda())
    trainer.set_window_select(GPUWindowSelect().cuda())
    trainer.set_warp(GPUWarp().cuda())
    trainer.set_augment(
        GPUBatchAugment(
            brightness=augment_cfg.brightness,
            contrast=augment_cfg.contrast,
            saturation=augment_cfg.saturation,
            hue=augment_cfg.hue,
            noise_ampl=augment_cfg.noise_ampl,
            rgb_mean=augment_cfg.rgb_mean,
            rgb_std=augment_cfg.rgb_std,
        ).cuda()
    )

    return trainer
