import optuna
from transformers import TrainerCallback

from constants import FLOW_MMA_KEY


class OptunaPruningCallback(TrainerCallback):
    """Per-epoch pruning callback for Optuna trials.

    Reports eval_flow_MMA to the trial after each flow-dataset evaluation
    pass, then raises TrialPruned when the trial should be stopped early.
    The runner already skips the final HPatches eval on the trial path, so
    raising TrialPruned here is sufficient to bypass it cleanly.
    """

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Only report when the flow eval key is present. The HPatches probe
        # eval runs on the same callback but produces "eval_MMA" (no flow
        # prefix), so FLOW_MMA_KEY will be absent — skip silently to avoid
        # reporting a wrong value or crashing.
        if FLOW_MMA_KEY not in metrics:
            return

        # Use the integer epoch as the step so Optuna's pruner reasons in
        # whole-epoch units. state.epoch is a float (e.g. 1.0, 2.0) after
        # each full epoch; rounding guards against floating-point noise.
        step = round(state.epoch)
        self.trial.report(metrics[FLOW_MMA_KEY], step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()
