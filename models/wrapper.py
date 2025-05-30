import torch
import torch.nn as nn

class ModelLossWrapper(nn.Module):
    def __init__(self, model, loss):
        super(ModelLossWrapper, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, **inputs):
        # Forward pass through the model
        output = self.model(imgs=[inputs.pop('img_a'),inputs.pop('img_b')])
        
        # Compute the loss
        allvars = inputs | output
        loss, details = self.loss(**allvars)
        
        # Return both the model outputs and the loss
        return output | {'loss': loss, 'loss_details': details}