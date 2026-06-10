from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelCfg:
    name: str = "Quad_L2Net_ConfCFS"
    # Quad_L2Net_ConfCFS / Quad_L2Net params
    dim: int = 128
    mchan: int = 4
    relu22: bool = False
    # ConvNeXtV2 params
    convnext_scale: str = "tiny"


@dataclass
class OptimCfg:
    optim: str = "schedule_free_adamw"
    lr_scheduler_type: str = "constant"
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 5e-4
    max_grad_norm: float = 0.0


@dataclass
class TrainerCfg:
    output_dir: str = "/workspace/outputs"
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 1
    num_train_epochs: int = 25
    eval_strategy: str = "epoch"
    save_strategy: str = "best"
    logging_steps: int = 100
    dataloader_num_workers: int = 4
    bf16: bool = True
    report_to: str = "none"
    seed: int = 42
    resume: bool = True
    batch_eval_metrics: bool = True


@dataclass
class SamplerCfg:
    ngh: int = 7
    subq: int = -8
    subd: int = 1
    pos_d: int = 3
    neg_d: int = 5
    border: int = 16
    subd_neg: int = -8
    maxpool_pos: bool = True


@dataclass
class LossTermCfg:
    weight: float = 1.0
    params: Any = field(default_factory=dict)


@dataclass
class LossCfg:
    reliability: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        weight=1.0, params={"base": 0.5, "nq": 20}
    ))
    cosim: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        weight=1.0, params={"N": 16}
    ))
    peaky: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        weight=1.0, params={"N": 16}
    ))


@dataclass
class AugmentCfg:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    noise_ampl: float = 25.0
    rgb_mean: Any = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_std: Any = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TransformCfg:
    # RandomScale(min_size, max_size, can_upscale) applied in ParametricTransform
    scale_min: int = 256
    scale_max: int = 1024
    scale_can_upscale: bool = True
    # RandomTilting(magnitude)
    tilt_magnitude: float = 0.5


@dataclass
class EvalCfg:
    # NonMaxSuppression thresholds (custom_trainer.py:110 — default ctor values)
    rel_thr: float = 0.7
    rep_thr: float = 0.7
    # extract_multiscale defaults from models/evaluation/utils.py:24-27
    scale_f: float = 1.189207115002721   # 2**0.25
    min_scale: float = 0.0
    max_scale: float = 1.0
    min_size: int = 256
    max_size: int = 1024
    # flow eval override: custom_trainer.py:149 uses min_size=192
    flow_min_size: int = 192


@dataclass
class RunCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    sampler: SamplerCfg = field(default_factory=SamplerCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    transform: TransformCfg = field(default_factory=TransformCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
