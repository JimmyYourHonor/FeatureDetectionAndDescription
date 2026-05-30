# from transformers import ConvNextV2Backbone
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2LayerNorm, ConvNextV2GRN, ConvNextV2DropPath, ACT2FN
from .patchnet import BaseNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNextV2Layer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = ACT2FN[config.hidden_act]
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

class ConvNextV2Stage(nn.Module):
    """ConvNeXTV2 stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`list[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=3, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextV2LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=stride, padding=stride),
            )
        elif in_channels != out_channels:
            self.downsampling_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[ConvNextV2Layer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states

class ConvNextV2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.tolist()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu").split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        dilation = 1
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=dilation,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            if i > 0:
                dilation *= 2
            prev_chs = out_chs
    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        for layer_module in self.stages:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class ConvNeXtV2(BaseNet):
    def __init__(self, model_scale='tiny'):
        super().__init__()
        if model_scale == 'nano':
            self.config = nano_config()
        if model_scale == 'tiny':
            self.config = tiny_config()
        elif model_scale == 'base':
            self.config = base_config()
        self.embbedding = nn.Sequential(
            nn.Conv2d(3, self.config.hidden_sizes[0], kernel_size=self.config.patch_size, stride=self.config.patch_size),
            ConvNextV2LayerNorm(self.config.hidden_sizes[0], eps=1e-6, data_format="channels_first"),
        )
        self.encoder = ConvNextV2Encoder(self.config)
        self.clf = nn.Conv2d(self.config.hidden_sizes[-1], 2, kernel_size=1)
        self.sal = nn.Conv2d(self.config.hidden_sizes[-1], 1, kernel_size=1)

    def forward_one(self, x):
        H, W = x.shape[2], x.shape[3]
        p = self.config.patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.embbedding(x)
        x = self.encoder(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)
    

def nano_config():
    return ConvNextV2Config(
        num_channels=3,
        patch_size=2,
        num_stages=4,
        hidden_sizes=[96, 96, 96, 96],
        depths=[2, 2, 2, 2],
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        image_size=192,
    )
def tiny_config():
    return ConvNextV2Config(
        num_channels=3,
        patch_size=2,
        num_stages=4,
        hidden_sizes=[128, 128, 128, 128],
        depths=[3, 3, 3, 3],
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        image_size=192,
    )
def base_config():
    return ConvNextV2Config(
        num_channels=3,
        patch_size=2,
        num_stages=4,
        hidden_sizes=[256, 256, 256, 256],
        depths=[3, 4, 6, 3],
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        image_size=192,
    )