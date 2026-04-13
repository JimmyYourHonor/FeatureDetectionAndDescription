import torch
import numpy as np
from transformers.trainer_utils import EvalPrediction
from typing import Mapping

n_feats = []
n_matches = []
err = {thr: 0 for thr in range(1, 16)}
total_pairs = 0


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


def _reset_state():
    global n_feats, n_matches, err, total_pairs
    n_feats.clear()
    n_matches.clear()
    for thr in range(1, 16):
        err[thr] = 0
    total_pairs = 0


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


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
    """
    global n_feats, n_matches, err, total_pairs
    predictions = evaluation_results.predictions
    labels = evaluation_results.label_ids

    if len(predictions) == 6:
        # HPatches format: one sequence, 5 pairs against the reference image
        keypoint_a = predictions[0][:, :2]
        descriptor_a = predictions[0][:, 2:]
        n_feats.append(keypoint_a.shape[0])
        for i in range(1, 6):
            keypoint_b = predictions[i][:, :2]
            descriptor_b = predictions[i][:, 2:]
            matches = mnn_matcher(descriptor_a, descriptor_b)
            if matches is None:
                continue
            n_matches.append(matches.shape[0])
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
                err[thr] += torch.mean((dist <= thr).float())
        total_pairs += 5

    else:
        # Flow format: one image pair, GT correspondence from aflow
        keypoint_a = predictions[0][:, :2]
        descriptor_a = predictions[0][:, 2:]
        keypoint_b = predictions[1][:, :2]
        descriptor_b = predictions[1][:, 2:]
        n_feats.append(keypoint_a.shape[0])
        matches = mnn_matcher(descriptor_a, descriptor_b)
        if matches is not None:
            n_matches.append(matches.shape[0])
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
                    err[thr] += torch.mean((dist <= thr).float())
        total_pairs += 1

    if compute_result:
        if total_pairs == 0:
            _reset_state()
            return {"MMA": 0.0}
        mean_matching_acc = sum(
            _to_float(err[thr]) / total_pairs for thr in range(1, 16)
        ) / 15
        result = {
            **{f"error_{thr}": _to_float(err[thr]) / total_pairs for thr in range(1, 16)},
            "MMA": mean_matching_acc,
            "avg_matches": (
                torch.sum(torch.tensor(n_matches)).item() / total_pairs
                if n_matches else 0.0
            ),
            "avg_feats": (
                torch.mean(torch.tensor(n_feats, dtype=torch.float32)).item()
                if n_feats else 0.0
            ),
            "min_feats": (
                torch.min(torch.tensor(n_feats, dtype=torch.float32)).item()
                if n_feats else 0.0
            ),
            "max_feats": (
                torch.max(torch.tensor(n_feats, dtype=torch.float32)).item()
                if n_feats else 0.0
            ),
        }
        _reset_state()
        return result
