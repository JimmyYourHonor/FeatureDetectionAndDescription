from transformers import Trainer

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