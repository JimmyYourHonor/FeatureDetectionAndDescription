import torch
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_outputs import ModelOutput
from typing import Mapping

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

    import pdb
    pdb.set_trace()
