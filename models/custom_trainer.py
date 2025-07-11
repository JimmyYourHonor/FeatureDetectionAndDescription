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
        detector = NonMaxSuppression()

        for i in range(1,7):
            img = inputs[f'{i}.ppm']
            xys, desc, scores = extract_multiscale( model, img, detector )
            xys = xys.cpu().numpy()
            desc = desc.cpu().numpy()
            scores = scores.cpu().numpy()
            idxs = scores.argsort()[-5000 or None:]
            import pdb
            pdb.set_trace()
        # if len(logits) == 1:
        #     logits = logits[0]

        # return (loss, logits, labels)