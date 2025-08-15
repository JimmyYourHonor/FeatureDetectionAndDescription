import wandb
from transformers.integrations import WandbCallback

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
            xs = [i for i in range(1, 16)]
            ys = [metrics[f"eval_error_{i}"] for i in range(1, 16)]
            table = wandb.Table(data=[[x, y] for x, y in zip(xs, ys)], columns=["Thresholds", "Precision"])
            self._wandb.log({"Precision vs Recall curve": wandb.plot.line(
                table, "Thresholds", "Precision",
                title="Precision vs Recall Curve"
            )})