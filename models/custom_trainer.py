import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional, Union, Any
from models.evaluation.utils import NonMaxSuppression, extract_multiscale

class CustomTrainer(Trainer):
    def set_loss(self, loss):
        """Set the loss function for the trainer."""
        self.loss = loss

    def set_augment(self, augment):
        """Set the GPU batch augment module (see preprocessing.GPUBatchAugment).

        The augment is invoked on `img_a`/`img_b` inputs to convert uint8 →
        normalized float32 with optional color/noise augmentation. Required
        for training; eval calls it with training=False.
        """
        self.augment = augment

    def set_warp(self, warp):
        """Set the GPU batch warp module (see preprocessing.GPUWarp).

        The warp is invoked on `src_a`/`src_b` (raw uint8 sources) plus the
        per-sample `M_a`/`M_b` homographies emitted by ParametricTransform.
        It produces the cropped uint8 `img_a`/`img_b` consumed by the augment.
        """
        self.warp = warp

    def _apply_warp(self, inputs):
        """Render img_a/img_b from src_a/src_b in-place when present."""
        if 'src_a' in inputs and 'src_b' in inputs:
            src_a = inputs.pop('src_a')
            src_b = inputs.pop('src_b')
            M_a = inputs.pop('M_a')
            M_b = inputs.pop('M_b')
            inputs['img_a'], inputs['img_b'] = self.warp(src_a, src_b, M_a, M_b)
        return inputs

    def _apply_augment(self, inputs, training: bool):
        """Run the GPU augment on img_a/img_b in-place if both are present."""
        if 'img_a' in inputs and 'img_b' in inputs:
            inputs['img_a'], inputs['img_b'] = self.augment(
                inputs['img_a'], inputs['img_b'], training=training
            )
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # GPU-side warp (src_a/src_b → img_a/img_b) then augment + normalization
        inputs = self._apply_warp(inputs)
        inputs = self._apply_augment(inputs, training=True)
        # forward pass
        outputs = model(imgs=[inputs.pop('img_a'), inputs.pop('img_b')])
        # Compute the loss
        allvars = inputs | outputs
        loss, details = self.loss(**allvars)

        # If return_outputs is True, we return the outputs as well
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Handles two dataset formats:

        HPatches format (keys '1.ppm' … '6.ppm'):
            Extracts features from all 6 images, returns list of 6 feature
            tensors and the corresponding homography matrices as labels.

        Flow format (keys 'img_a', 'img_b', 'aflow'):
            Extracts features from both images, returns list of 2 feature
            tensors and the aflow tensor as labels.

        Each feature tensor has shape (N, 3 + D): columns are [x, y, scale, descriptors].
        """
        model.eval()
        detector = NonMaxSuppression()

        if '1.ppm' in inputs:
            # HPatches format: 6 images per sequence
            logits = []
            labels = []
            for i in range(1, 7):
                img = inputs[f'{i}.ppm']
                xys, desc, scores = extract_multiscale(model, img, detector)
                idxs = scores.argsort()[-5000 or None:]
                xys = xys[idxs]
                desc = desc[idxs]
                result = torch.cat((xys, desc), dim=1)
                logits.append(result)
                labels.append(
                    inputs[f'h_1_{i}'].to(torch.float32)
                    if i != 1
                    else torch.eye(3, device=img.device).unsqueeze(0)
                )
            labels = torch.cat(labels, dim=0)
            return (None, logits, labels)

        else:
            # Flow format: one image pair from the training distribution.
            # Inputs are raw sources (src_a/src_b + M_a/M_b) emitted by
            # ParametricTransform. Run warp then augment with training=False
            # (no noise / color jitter at eval).
            inputs = self._apply_warp(inputs)
            inputs = self._apply_augment(inputs, training=False)
            img_a = inputs['img_a']  # (1, 3, H, W)
            img_b = inputs['img_b']  # (1, 3, H, W)
            aflow = inputs['aflow']  # (1, 2, H, W)

            logits = []
            for img in [img_a, img_b]:
                xys, desc, scores = extract_multiscale(model, img, detector)
                idxs = scores.argsort()[-5000 or None:]
                xys = xys[idxs]
                desc = desc[idxs]
                result = torch.cat((xys, desc), dim=1)
                logits.append(result)

            # Remove batch dim from aflow so compute_metrics receives (2, H, W)
            return (None, logits, aflow.squeeze(0))
