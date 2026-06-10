# Centralized metric-key constants.
#
# HF Trainer prefixes eval metrics with "eval_<dataset_key>_" when
# eval_dataset is a dict. The flow eval set is keyed "flow", so its
# metrics become "eval_flow_MMA", "eval_flow_error_1", etc.
# The HPatches final eval uses a direct trainer.evaluate() call (no dict
# key), so its metrics have no dataset prefix: "eval_MMA".
#
# metric_for_best_model takes the UNPREFIXED form (Trainer prepends "eval_"
# before resolving, but it expects just the key after "eval_").

FLOW_MMA_KEY = "eval_flow_MMA"
FLOW_PREFIX = "eval_flow_"
HPATCHES_MMA_KEY = "eval_MMA"

# Unprefixed form used by metric_for_best_model in TrainingArguments.
FLOW_MMA_UNPREFIXED = "flow_MMA"
