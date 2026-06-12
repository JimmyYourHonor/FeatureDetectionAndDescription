"""Regression tests: default.yaml + builders reproduce prior hardcoded train.py values.

These tests do NOT require GPU, HF Hub access, or accelerate. They only construct
config, sampler, loss, and EvalCfg objects — not datasets or trainers.
"""

import pytest

from config import load_config
from build import build_sampler, build_loss
from constants import FLOW_MMA_UNPREFIXED
from config.schema import EvalCfg


CONFIG_PATH = "config/default.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


class TestBuildTrainingArgs:
    """build_training_args(cfg) reproduces prior hardcoded TrainingArguments values.

    TrainingArguments requires accelerate at construction, so we assert directly
    against the cfg fields that build_training_args reads — not against a live
    TrainingArguments instance. This is sufficient to pin the baseline.
    """

    def test_learning_rate(self, cfg):
        assert cfg.optim.learning_rate == 5e-4

    def test_per_device_train_batch_size(self, cfg):
        assert cfg.trainer.per_device_train_batch_size == 16

    def test_num_train_epochs(self, cfg):
        assert cfg.trainer.num_train_epochs == 25

    def test_optim(self, cfg):
        assert cfg.optim.optim == "schedule_free_adamw"

    def test_lr_scheduler_type(self, cfg):
        assert cfg.optim.lr_scheduler_type == "constant"

    def test_warmup_steps(self, cfg):
        assert cfg.optim.warmup_steps == 500

    def test_weight_decay(self, cfg):
        assert cfg.optim.weight_decay == 5e-4

    def test_max_grad_norm(self, cfg):
        assert cfg.optim.max_grad_norm == 0.0

    def test_bf16(self, cfg):
        assert cfg.trainer.bf16 is True

    def test_eval_strategy(self, cfg):
        assert cfg.trainer.eval_strategy == "epoch"

    def test_save_strategy(self, cfg):
        assert cfg.trainer.save_strategy == "best"

    def test_seed(self, cfg):
        assert cfg.trainer.seed == 42

    def test_metric_for_best_model(self, cfg):
        # build_training_args passes FLOW_MMA_UNPREFIXED to metric_for_best_model
        assert FLOW_MMA_UNPREFIXED == "flow_MMA"

    def test_greater_is_better(self):
        # build_training_args hardcodes greater_is_better=True
        # (not stored in cfg; checked against the build.py source intent)
        from build import build_training_args
        # Inspect the kwarg default in the source rather than instantiating
        # TrainingArguments (which requires accelerate).
        import inspect
        src = inspect.getsource(build_training_args)
        assert "greater_is_better=True" in src

    def test_batch_eval_metrics(self, cfg):
        # train.py:97 — required for MetricAccumulator per-batch contract and correct MMA
        assert cfg.trainer.batch_eval_metrics is True

    def test_report_to(self, cfg):
        assert cfg.trainer.report_to == "none"

    def test_output_dir(self, cfg):
        assert cfg.trainer.output_dir == "/workspace/outputs"

    def test_load_best_model_at_end(self, cfg):
        # train.py:104 — load_best_model_at_end was commented out in the baseline,
        # so the default must be False.  build_training_args passes it through to
        # TrainingArguments; keeping it False preserves identical behavior to the
        # pre-refactor train.py while making it config-overridable if 5.5.4 requires it.
        assert cfg.trainer.load_best_model_at_end is False


class TestBuildSampler:
    """build_sampler(cfg) reproduces prior hardcoded NghSampler2 constructor values."""

    @pytest.fixture(scope="class")
    def sampler(self, cfg):
        return build_sampler(cfg)

    def test_type(self, sampler):
        from models.sampler.sampler import NghSampler2
        assert isinstance(sampler, NghSampler2)

    def test_ngh(self, sampler):
        assert sampler.ngh == 7

    def test_sub_q(self, sampler):
        # NghSampler2 stores subq as self.sub_q
        assert sampler.sub_q == -8

    def test_sub_d(self, sampler):
        # NghSampler2 stores subd as self.sub_d
        assert sampler.sub_d == 1

    def test_pos_d(self, sampler):
        assert sampler.pos_d == 3

    def test_neg_d(self, sampler):
        assert sampler.neg_d == 5

    def test_border(self, sampler):
        assert sampler.border == 16

    def test_sub_d_neg(self, sampler):
        # NghSampler2 stores subd_neg as self.sub_d_neg
        assert sampler.sub_d_neg == -8

    def test_maxpool_pos(self, sampler):
        assert sampler.maxpool_pos is True

    def test_invariant_pos_d_lt_neg_d_lte_ngh(self, cfg):
        s = cfg.sampler
        assert s.pos_d < s.neg_d <= s.ngh


class TestBuildLoss:
    """build_loss(cfg, sampler) reproduces prior hardcoded MultiLoss values."""

    @pytest.fixture(scope="class")
    def sampler(self, cfg):
        return build_sampler(cfg)

    @pytest.fixture(scope="class")
    def loss(self, cfg, sampler):
        return build_loss(cfg, sampler)

    def test_type(self, loss):
        from models.loss.losses import MultiLoss
        assert isinstance(loss, MultiLoss)

    def test_three_loss_terms(self, loss):
        assert len(loss.losses) == 3
        assert len(loss.weights) == 3

    def test_reliability_weight(self, loss):
        assert loss.weights[0] == 1.0

    def test_cosim_weight(self, loss):
        assert loss.weights[1] == 1.0

    def test_peaky_weight(self, loss):
        assert loss.weights[2] == 1.0

    def test_reliability_base(self, loss):
        from models.loss.reliability_loss import ReliabilityLoss
        rel = loss.losses[0]
        assert isinstance(rel, ReliabilityLoss)
        assert rel.base == 0.5

    def test_reliability_nq(self, loss):
        rel = loss.losses[0]
        assert rel.aploss.nq == 20

    def test_cosim_N(self, loss):
        from models.loss.repeatability_loss import CosimLoss
        cos = loss.losses[1]
        assert isinstance(cos, CosimLoss)
        # CosimLoss stores N via nn.Unfold(kernel_size=N, ...)
        # nn.Unfold keeps kernel_size as the raw scalar when constructed with an int
        assert cos.patches.kernel_size == 16

    def test_peaky_N(self, loss):
        from models.loss.repeatability_loss import PeakyLoss
        pky = loss.losses[2]
        assert isinstance(pky, PeakyLoss)
        assert pky.maxpool.kernel_size == 17  # N+1 = 17

    def test_cfg_params_match(self, cfg):
        assert cfg.loss.reliability.params["base"] == 0.5
        assert cfg.loss.reliability.params["nq"] == 20
        assert cfg.loss.cosim.params["N"] == 16
        assert cfg.loss.peaky.params["N"] == 16


class TestEvalCfgDefaults:
    """EvalCfg defaults match the prior hardcoded eval knobs in custom_trainer.py."""

    @pytest.fixture(scope="class")
    def eval_cfg(self, cfg):
        return cfg.eval

    def test_type(self, eval_cfg):
        assert isinstance(eval_cfg, EvalCfg)

    def test_rel_thr(self, eval_cfg):
        # NonMaxSuppression default was 0.7 (models/evaluation/utils.py:7)
        assert eval_cfg.rel_thr == 0.7

    def test_rep_thr(self, eval_cfg):
        # NonMaxSuppression default was 0.7 (models/evaluation/utils.py:8)
        assert eval_cfg.rep_thr == 0.7

    def test_scale_f(self, eval_cfg):
        # extract_multiscale default was 2**0.25 (models/evaluation/utils.py:24)
        assert abs(eval_cfg.scale_f - 2 ** 0.25) < 1e-12

    def test_min_scale(self, eval_cfg):
        assert eval_cfg.min_scale == 0.0

    def test_max_scale(self, eval_cfg):
        assert eval_cfg.max_scale == 1.0

    def test_min_size(self, eval_cfg):
        assert eval_cfg.min_size == 256

    def test_max_size(self, eval_cfg):
        assert eval_cfg.max_size == 1024

    def test_flow_min_size(self, eval_cfg):
        # custom_trainer.py:149 used min_size=192 for flow eval
        assert eval_cfg.flow_min_size == 192
