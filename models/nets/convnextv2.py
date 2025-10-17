from transformers import ConvNextV2Backbone
from .patchnet import BaseNet

class ConvNeXtV2(BaseNet):
    def __init__(self, model_scale='tiny'):
        super().__init__()
        if model_scale == 'nano':
            self.backbone = ConvNextV2Backbone.from_pretrained("facebook/convnextv2-nano-22k-224")
        if model_scale == 'tiny':
            self.backbone = ConvNextV2Backbone.from_pretrained("facebook/convnextv2-tiny-22k-224")
        elif model_scale == 'base':
            self.backbone = ConvNextV2Backbone.from_pretrained("facebook/convnextv2-base-22k-224")
    def forward_one(self, x):
        output = self.backbone(x)
        return self.normalize(output.feature_maps[-1])