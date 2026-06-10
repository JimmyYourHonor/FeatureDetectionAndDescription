import torch
import numpy as np
from transformers.trainer_utils import EvalPrediction
from typing import Callable, Mapping, Tuple


def mnn_matcher(descriptors_a, descriptors_b):
    if descriptors_a.shape[0] == 0 or descriptors_b.shape[0] == 0:
        return None
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t()


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


class MetricAccumulator:
    """Per-run accumulator for MMA and related feature-matching statistics.

    Usage for Optuna runs (T-1.6):
        compute_fn, acc = make_compute_metrics()
        trainer = build_trainer(..., compute_metrics=compute_fn)
        trainer.train()
        # acc holds no cross-trial state; the next trial gets a fresh pair.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.n_feats = []
        self.n_matches = []
        self.err = {thr: 0 for thr in range(1, 16)}
        self.total_pairs = 0

    def update(self, evaluation_results: EvalPrediction) -> None:
        predictions = evaluation_results.predictions
        labels = evaluation_results.label_ids

        if len(predictions) == 6:
            # HPatches format: one sequence, 5 pairs against the reference image
            keypoint_a = predictions[0][:, :2]
            descriptor_a = predictions[0][:, 2:]
            self.n_feats.append(keypoint_a.shape[0])
            for i in range(1, 6):
                keypoint_b = predictions[i][:, :2]
                descriptor_b = predictions[i][:, 2:]
                matches = mnn_matcher(descriptor_a, descriptor_b)
                if matches is None:
                    continue
                self.n_matches.append(matches.shape[0])
                homography = labels[i]
                pos_a = keypoint_a[matches[:, 0], :2]
                pos_a_h = torch.cat(
                    [pos_a, torch.ones([matches.shape[0], 1], device=pos_a.device)], dim=1
                )
                pos_b_proj_h = torch.transpose(
                    torch.matmul(homography, torch.transpose(pos_a_h, 0, 1)), 0, 1
                )
                pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:].clamp(min=1e-8)
                pos_b = keypoint_b[matches[:, 1], :2]
                dist = torch.sqrt(torch.sum((pos_b - pos_b_proj) ** 2, dim=1))
                for thr in range(1, 16):
                    self.err[thr] += torch.mean((dist <= thr).float())
            self.total_pairs += 5

        else:
            # Flow format: one image pair, GT correspondence from aflow
            keypoint_a = predictions[0][:, :2]
            descriptor_a = predictions[0][:, 2:]
            keypoint_b = predictions[1][:, :2]
            descriptor_b = predictions[1][:, 2:]
            self.n_feats.append(keypoint_a.shape[0])
            matches = mnn_matcher(descriptor_a, descriptor_b)
            if matches is not None:
                self.n_matches.append(matches.shape[0])
                aflow = labels  # (2, H, W)
                if isinstance(aflow, np.ndarray):
                    aflow = torch.from_numpy(aflow)
                _, H, W = aflow.shape
                pos_a = keypoint_a[matches[:, 0], :2]
                x = pos_a[:, 0].long().clamp(0, W - 1)
                y = pos_a[:, 1].long().clamp(0, H - 1)
                x_gt = aflow[0, y, x]
                y_gt = aflow[1, y, x]
                # aflow encodes NaN for masked (invalid) regions — skip those
                valid = ~(torch.isnan(x_gt) | torch.isnan(y_gt))
                if valid.any():
                    pos_b_gt = torch.stack([x_gt[valid], y_gt[valid]], dim=1)
                    pos_b = keypoint_b[matches[valid, 1], :2]
                    dist = torch.sqrt(torch.sum((pos_b - pos_b_gt) ** 2, dim=1))
                    for thr in range(1, 16):
                        self.err[thr] += torch.mean((dist <= thr).float())
            self.total_pairs += 1

    def result(self) -> dict:
        if self.total_pairs == 0:
            self.reset()
            return {"MMA": 0.0}
        mean_matching_acc = sum(
            _to_float(self.err[thr]) / self.total_pairs for thr in range(1, 16)
        ) / 15
        result = {
            **{f"error_{thr}": _to_float(self.err[thr]) / self.total_pairs for thr in range(1, 16)},
            "MMA": mean_matching_acc,
            "avg_matches": (
                torch.sum(torch.tensor(self.n_matches)).item() / self.total_pairs
                if self.n_matches else 0.0
            ),
            "avg_feats": (
                torch.mean(torch.tensor(self.n_feats, dtype=torch.float32)).item()
                if self.n_feats else 0.0
            ),
            "min_feats": (
                torch.min(torch.tensor(self.n_feats, dtype=torch.float32)).item()
                if self.n_feats else 0.0
            ),
            "max_feats": (
                torch.max(torch.tensor(self.n_feats, dtype=torch.float32)).item()
                if self.n_feats else 0.0
            ),
        }
        self.reset()
        return result


# Module-level default instance — used by the top-level compute_metrics /
# _reset_state pair so that existing callers (train.py, CustomTrainer, tests)
# require zero changes.
_default_accumulator = MetricAccumulator()


def _reset_state() -> None:
    """Reset the module-level default accumulator. Tests call this via the
    autouse fixture; it is equivalent to _default_accumulator.reset()."""
    _default_accumulator.reset()


def compute_metrics(
    evaluation_results: EvalPrediction, compute_result: bool = False
) -> Mapping[str, float]:
    """
    Metric computation for feature matching tasks.
    Compute the mean matching accuracy (MMA): the average percentage of correct
    matches across pixel error thresholds 1–15 px.

    Supports two input formats:

    HPatches format (len(predictions) == 6):
        predictions: list of 6 tensors (N_i, 3+D) — one per image in sequence
        label_ids:   tensor (6, 3, 3) — homography matrices H_1_i
        Computes MMA for the 5 pairs (image 1 vs images 2–6).

    Flow format (len(predictions) == 2):
        predictions: list of 2 tensors (N_i, 3+D) — img_a and img_b features
        label_ids:   tensor (2, H, W) — absolute optical flow (aflow)
        Computes MMA for the single pair using aflow for GT correspondence.

    Each tensor in predictions has columns: [x, y, scale, desc_0, ..., desc_D].

    Delegates to the module-level _default_accumulator. For per-trial isolation
    in Optuna runs, use make_compute_metrics() instead.
    """
    _default_accumulator.update(evaluation_results)
    if compute_result:
        return _default_accumulator.result()
    return None


def make_compute_metrics() -> Tuple[Callable, MetricAccumulator]:
    """Factory for per-run metric isolation (used by T-1.6 / runner.py).

    Returns a (compute_fn, accumulator) pair backed by a fresh MetricAccumulator.
    Pass compute_fn to CustomTrainer/build_trainer as the compute_metrics argument.
    The accumulator is exposed so runner.py can call acc.reset() on entry and
    inspect acc after training if needed.

    Contract for T-1.6:
        compute_fn, acc = make_compute_metrics()
        # ... build trainer with compute_metrics=compute_fn ...
        trainer.train()
        # Next trial: call make_compute_metrics() again — no shared state.
    """
    acc = MetricAccumulator()

    def _compute_metrics(
        evaluation_results: EvalPrediction, compute_result: bool = False
    ) -> Mapping[str, float]:
        acc.update(evaluation_results)
        if compute_result:
            return acc.result()
        return None

    return _compute_metrics, acc
