import wandb
from transformers.integrations import WandbCallback
from constants import FLOW_PREFIX

class EvalCallback(WandbCallback):
    """
    A callback to evaluate the model during training.
    And reports to Weights & Biases (wandb).
    """
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Called at the end of each evaluation.
        Plot the Precision vs Recall curve to Weights & Biases (wandb) on the last epoch.
        """
        # The Trainer automatically logs scalar metrics when report_to="wandb".
        # This callback is for adding custom plots.

        # Only plot on the final evaluation at the end of training.
        if state.epoch >= state.num_train_epochs:
            # With multiple eval datasets, metrics are prefixed per set
            # (e.g. eval_flow_error_1). Plot the flow val curve; skip the
            # per-dataset calls (e.g. hpatches) that lack these keys.
            prefix = FLOW_PREFIX if f"{FLOW_PREFIX}error_1" in metrics else "eval_"
            if f"{prefix}error_1" not in metrics:
                return
            xs = [i for i in range(1, 16)]
            ys = [metrics[f"{prefix}error_{i}"] for i in range(1, 16)]
            table = wandb.Table(data=[[x, y] for x, y in zip(xs, ys)], columns=["Thresholds", "Precision"])
            self._wandb.log({"Precision vs Recall curve": wandb.plot.line(
                table, "Thresholds", "Precision",
                title="Precision vs Recall Curve"
            )})