import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional, Union, Any
from models.evaluation.utils import NonMaxSuppression, extract_multiscale

class CustomTrainer(Trainer):
    def set_loss(self, loss):
        """Set the loss function for the trainer."""
        self.loss = loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # forward pass
        outputs = model(imgs=[inputs.pop('img_a'),inputs.pop('img_b')])
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
        Perform an evaluation step on `model` using `inputs`. Overide for multiscale evaluation.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        model.eval()
        detector = NonMaxSuppression(rel_thr=0.4, rep_thr=0.4)
        logits = []
        labels = []
        for i in range(1,7):
            img = inputs[f'{i}.ppm']
            xys, desc, scores = extract_multiscale( model, img, detector )
            idxs = scores.argsort()[-5000 or None:]
            xys = xys[idxs]
            desc = desc[idxs]
            scores = scores[idxs]
            result = torch.cat((xys, desc, scores.unsqueeze(1)), dim=1)
            logits.append(result)
            labels.append(inputs[f'h_1_{i}'] if i != 1 else torch.eye(3, device=img.device).unsqueeze(0))
        logits = torch.stack(logits, dim=0)
        labels = torch.stack(labels, dim=0)

        return (None, logits, labels)