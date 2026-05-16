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
        per-sample `M_a`/`M_b` homographies produced by GPUWindowSelect.
        It produces the cropped uint8 `img_a`/`img_b` consumed by the augment.
        """
        self.warp = warp

    def set_window_select(self, window_select):
        """Set the GPU window-selection module (see preprocessing.GPUWindowSelect).

        Consumes the lightweight per-sample chain matrices emitted by
        ParametricTransform (`sa2ia`/`sb2ib`/`M_ab`/`img_size`/`mode` plus
        flow tensors), runs the trial-loop crop selection on GPU, and
        produces the per-sample `M_a`/`M_b`/`aflow`/`mask` tensors that
        GPUWarp + the loss expect.
        """
        self.window_select = window_select

    def _apply_window_select(self, inputs):
        """Run window selection in-place when ParametricTransform outputs are present."""
        if 'sa2ia' in inputs:
            M_a, M_b, aflow, mask = self.window_select(
                inputs.pop('sa2ia'), inputs.pop('sb2ib'), inputs.pop('M_ab'),
                inputs.pop('img_size'), inputs.pop('mode'),
                inputs.pop('aflow_full'), inputs.pop('mask_full'),
            )
            inputs['M_a'] = M_a
            inputs['M_b'] = M_b
            inputs['aflow'] = aflow
            inputs['mask'] = mask
        return inputs

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
        # GPU-side window selection → warp → augment + normalization
        inputs = self._apply_window_select(inputs)
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
            # Inputs are the lightweight per-sample geometric chain emitted
            # by ParametricTransform. Run window selection → warp → augment
            # with training=False (no noise / color jitter at eval).
            inputs = self._apply_window_select(inputs)
            inputs = self._apply_warp(inputs)
            inputs = self._apply_augment(inputs, training=False)
            img_a = inputs['img_a']  # (1, 3, H, W)
            img_b = inputs['img_b']  # (1, 3, H, W)
            aflow = inputs['aflow']  # (1, 2, H, W)

            logits = []
            for img in [img_a, img_b]:
                # min_size=0, min_scale=1.0: single-scale extraction at native
                # resolution. The default min_size=256 would skip 192×192 crops.
                xys, desc, scores = extract_multiscale(
                    model, img, detector, min_size=192
                )
                idxs = scores.argsort()[-5000 or None:]
                xys = xys[idxs]
                desc = desc[idxs]
                result = torch.cat((xys, desc), dim=1)
                logits.append(result)

            # Remove batch dim from aflow so compute_metrics receives (2, H, W)
            return (None, logits, aflow.squeeze(0))
