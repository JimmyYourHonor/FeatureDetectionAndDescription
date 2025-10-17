from transformers import ConvNextV2Backbone
from .patchnet import BaseNet
import torch.nn as nn

class ConvNeXtV2(BaseNet):
    def __init__(self, model_scale='tiny'):
        super().__init__()
        if model_scale == 'nano':
            self.backbone = ConvNextV2Backbone.from_pretrained("facebook/convnextv2-nano-22k-224")
        if model_scale == 'tiny':
            self.backbone = ConvNextV2Backbone.from_pretrained("facebook/convnextv2-tiny-22k-224")
        elif model_scale == 'base':
            self.backbone = ConvNextV2Backbone.from_pretrained("facebook/convnextv2-base-22k-224")
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
    def forward_one(self, x):
        output = self.backbone(x).feature_maps[-1]
        ureliability = self.clf(output**2)
        urepeatability = self.sal(output**2)
        return self.normalize(output, ureliability, urepeatability)