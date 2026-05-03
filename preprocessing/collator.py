"""Data collator for ParametricTransform.

`ParametricTransform` emits raw uint8 source images at variable resolution
(``src_a``/``src_b``), so they cannot be ``torch.stack``-ed across the batch.
This collator keeps those keys as Python lists and stacks every other tensor
key into a normal ``(B, ...)`` batch dimension.

Also passes through cleanly for the HPatches eval format (which has no
``src_a``/``src_b`` keys), letting it serve as the single ``data_collator``
on ``TrainingArguments`` for both training and evaluation datasets.
"""

import torch


_LIST_KEYS = ('src_a', 'src_b')


def parametric_collator(features):
    """Collate a list of per-sample dicts into a single batch dict."""
    batch = {}
    for k in features[0].keys():
        vals = [f[k] for f in features]
        if k in _LIST_KEYS:
            batch[k] = vals
        elif isinstance(vals[0], torch.Tensor):
            batch[k] = torch.stack(vals, dim=0)
        else:
            batch[k] = vals
    return batch
