import torch
from transformers.trainer_utils import EvalPrediction
from typing import Mapping

n_feats = []
n_matches = []
err = {thr: 0 for thr in range(1,16)}
total_pairs = 0

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t()

def compute_metrics(
    evaluation_results: EvalPrediction, compute_result: bool = False
) -> Mapping[str, float]:
    """
    Metric computation for feature matching tasks.
    Compute the mean matching accuracy which is the average percentage of correct matches 
    in an image pair considering multiple pixel error thresholds.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predications = evaluation_results.predictions
    labels = evaluation_results.label_ids
    keypoints = predications[:, :2]
    descriptors = predications[:, 3:131]
    keypoint_a = keypoints[0]
    descriptor_a = descriptors[0]
    n_feats.append(keypoint_a.shape[0])
    for i in range(1, 6):
        keypoint_b = keypoints[i]
        descriptor_b = descriptors[i]
        matches = mnn_matcher(descriptor_a, descriptor_b)
        if matches.shape[0] == 0:
            continue
        n_matches.append(matches.shape[0])
        homography = labels[i]
        pos_a = keypoint_a[matches[:, 0], : 2]
        pos_a_h = torch.cat([pos_a, torch.ones([matches.shape[0], 1], device=pos_a.device)], axis=1)
        pos_b_proj_h = torch.transpose(
            torch.matmul(homography, torch.transpose(pos_a_h, 0, 1)), 0, 1)
        pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:].clamp(min=1e-8)

        pos_b = keypoint_b[matches[:, 1], : 2]

        dist = torch.sqrt(torch.sum((pos_b - pos_b_proj) ** 2, dim=1))
        for thr in range(1,16):
            err[thr] += torch.mean(dist <= thr)
    total_pairs += 5  # 5 pairs per sequence
    if compute_result:
        mean_matching_acc = 0
        for thr in range(1, 16):
            mean_matching_acc += err[thr] / total_pairs
        mean_matching_acc /= 15
        return {
            **{f"error_{thr}": err[thr] / total_pairs for thr in range(1, 16)},
            "MMA": mean_matching_acc,
            "avg_matches": torch.sum(torch.tensor(n_matches)) / total_pairs,
            "avg_feats": torch.mean(torch.tensor(n_feats)),
            "min_feats": torch.min(torch.tensor(n_feats)),
            "max_feats": torch.max(torch.tensor(n_feats)),
        }