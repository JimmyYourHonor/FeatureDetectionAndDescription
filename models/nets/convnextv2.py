from transformers import ConvNextV2Backbone
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2LayerNorm, ConvNextV2Layer
from .patchnet import BaseNet
import torch
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

        # output all features
        self.backbone.out_features = self.backbone.stage_names
        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self.backbone.out_features, self.backbone.channels):
            hidden_states_norms[stage] = ConvNextV2LayerNorm(num_channels, data_format="channels_first")
        self.backbone.hidden_states_norms = nn.ModuleDict(hidden_states_norms)
        # initialize weights and apply final processing
        self.backbone.post_init()

        self.decoder = UnetDecoder(self.backbone.config, drop_path=0.1)

        # reliability classifier
        self.clf = nn.Conv2d(self.backbone.config.hidden_sizes[0], 2, kernel_size=1)
        # repeatability classifier
        self.sal = nn.Conv2d(self.backbone.config.hidden_sizes[0], 1, kernel_size=1)
    def forward_one(self, x):
        H, W = x.shape[2], x.shape[3]
        output = self.backbone(x).feature_maps
        output = self.decoder(output)
        output = nn.functional.interpolate(output, size=(H, W), mode='bilinear')
        ureliability = self.clf(output**2)
        urepeatability = self.sal(output**2)
        return self.normalize(output, ureliability, urepeatability)

class UnetDecoder(nn.Module):
    def __init__(self, config, drop_path=0.0):
        super(UnetDecoder, self).__init__()
        in_channels = [config.hidden_sizes[0]] + config.hidden_sizes
        self.layers = nn.ModuleList()
        for i in range(len(in_channels)-1, 1, -1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i]+in_channels[i-1], in_channels[i-1], kernel_size=1),
                    ConvNextV2Layer(config=config, dim=in_channels[i-1], drop_path=drop_path),
                )
            )

    def forward(self, features):
        x = features[-1]
        for i, layer in enumerate(self.layers):
            x_up = nn.functional.interpolate(x, size=features[-i-2].shape[2:], mode='bilinear')
            x = torch.cat([x_up, features[-i-2]], dim=1)
            x = layer(x)
        return x