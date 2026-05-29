import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchnet import BaseNet


class ViTDense(BaseNet):
    """Dense ViT backbone with DPT-style decoder.

    Emits the same three full-resolution tensors as the other BaseNet subclasses:
      descriptors   (B, out_dim, H, W)  — L2-normalized
      reliability   (B, 1, H, W)        — in [0,1]
      repeatability (B, 1, H, W)        — in [0,1]

    Architecture:
      _encode          — from-scratch ViT (ViTEmbeddings + ViTEncoder) with
                         interpolated positional encoding; taps hidden states at
                         4 configurable depths and drops the CLS token.
      _reassemble      — DPTReassembleStage turns (B, N_tokens, C) token sets
                         into image-like maps at strides 4, 8, 16, 32.
      _fuse            — intermediate 3×3 conv projection + DPTFeatureFusionStage
                         merges the 4 maps; output stride is /2 of the padded input.
      _final           — bilinear upsample from /2 to /1 (full padded resolution),
                         then _crop to exact (H, W).
      proj / clf / sal — 1×1 convs mapping decoder_channels → out_dim → heads.

    All learnable submodules are constructed eagerly in __init__ (via
    _build_encoder and _build_decoder) so HF Trainer can snapshot .parameters()
    before the first forward pass. Transformers imports are deferred inside those
    helpers to keep module-level import clean under torch 2.2.2.
    """

    def __init__(
        self,
        patch_size: int = 8,
        hidden_size: int = 160,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 4,
        out_dim: int = 160,
        tap_layers: tuple = (4, 10, 16, 22),
        decoder_channels: int = 160,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.out_dim = out_dim
        self.tap_layers = tap_layers
        self.decoder_channels = decoder_channels

        # Project decoder_channels → out_dim so descriptors match the contract
        self.proj = nn.Conv2d(decoder_channels, out_dim, kernel_size=1)

        # Heads operate on out_dim features (matching Quad_L2Net_ConfCFS / ConvNeXtV2)
        self.clf = nn.Conv2d(out_dim, 2, kernel_size=1)
        self.sal = nn.Conv2d(out_dim, 1, kernel_size=1)

        # Build encoder and decoder at construction time so all parameters are
        # registered before HF Trainer calls model.parameters() for the optimizer.
        self._build_encoder()
        self._build_decoder()

    # ------------------------------------------------------------------ #
    # Padding / cropping helpers                                           #
    # ------------------------------------------------------------------ #

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple:
        """Pad x spatially so H and W are multiples of patch_size.

        Returns (x_padded, (H_orig, W_orig), (pad_bottom, pad_right)).
        """
        p = self.patch_size
        _, _, H, W = x.shape
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (H, W), (pad_h, pad_w)

    def _crop(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Crop x back to the original (H, W)."""
        return x[:, :, :H, :W]

    # ------------------------------------------------------------------ #
    # Encoder construction + forward                                       #
    # ------------------------------------------------------------------ #

    def _build_encoder(self) -> None:
        """Construct ViTEmbeddings and ViTEncoder (called from __init__).

        Imports are deferred here so this module loads cleanly under torch 2.2.2,
        where transformers 5.5.4 disables all PyTorch-backed model classes.

        Assigning nn.Module instances to self.* via nn.Module.__setattr__
        registers them as proper submodules automatically.
        """
        from transformers.models.vit.modeling_vit import (
            ViTConfig,
            ViTEmbeddings,
            ViTEncoder,
        )

        cfg = ViTConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.hidden_size * 4,
            patch_size=self.patch_size,
            image_size=224,  # placeholder; overridden by interpolate_pos_encoding
            num_channels=3,
        )
        self.vit_embeddings = ViTEmbeddings(cfg)
        self.vit_encoder = ViTEncoder(cfg)

    def _encode(self, x: torch.Tensor) -> list:
        """Encode padded image into token sets at tap_layers depths.

        Args:
            x: (B, 3, Hp, Wp)  Hp and Wp are multiples of patch_size.

        Returns:
            list of len(tap_layers) tensors, each (B, N_tokens, hidden_size)
            where N_tokens = (Hp // patch_size) * (Wp // patch_size).

        CLS token: ViTEmbeddings prepends a CLS token; we drop index 0 before
        returning so _reassemble receives exactly N_tokens patch tokens.
        """
        hidden_states = self.vit_embeddings(x, interpolate_pos_encoding=True)

        tap_outputs = {}
        for depth, layer in enumerate(self.vit_encoder.layer):
            hidden_states = layer(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]  # unwrap BaseModelOutput
            if depth in self.tap_layers:
                tap_outputs[depth] = hidden_states[:, 1:, :]  # drop CLS token

        return [tap_outputs[d] for d in self.tap_layers]

    # ------------------------------------------------------------------ #
    # Decoder construction + forward                                       #
    # ------------------------------------------------------------------ #

    def _build_decoder(self) -> None:
        """Construct DPTReassembleStage, intermediate conv projections, and
        DPTFeatureFusionStage (called from __init__).

        Imports are deferred here; all constructed submodules are registered
        as nn.Module children via self.* assignment.

        DPTConfig fields used:
          hidden_size          — encoder token dimension (for DPTReassembleLayer projections)
          neck_hidden_sizes    — per-tap output channel dim after reassemble
          reassemble_factors   — [4, 2, 1, 0.5] → strides /4, /8, /16, /32
          readout_type         — "ignore": CLS-token split still happens inside
                                 DPTReassembleStage.forward (the code unconditionally
                                 does cls_token, hidden_state = hidden_state[:, 0],
                                 hidden_state[:, 1:]). We compensate by prepending a
                                 dummy zero CLS token in _reassemble so that split
                                 removes only the dummy, not a real patch token.
                                 "ignore" means the dummy cls_token value is never
                                 used in any computation.
          fusion_hidden_size   — decoder_channels (channel dim through fusion)
          is_hybrid            — False (pure ViT, no hybrid backbone)
          neck_ignore_stages   — [] (process all 4 taps)
          use_batch_norm_in_fusion_residual — False (LayerNorm-friendly)
          use_bias_in_fusion_residual       — True (bias on by default)
        """
        from transformers.models.dpt.modeling_dpt import (
            DPTConfig,
            DPTReassembleStage,
            DPTFeatureFusionStage,
        )

        # neck_hidden_sizes: all taps project to decoder_channels in the
        # reassemble stage, then the intermediate conv projects to fusion_hidden_size.
        # We set neck_hidden_sizes equal to decoder_channels for all 4 taps so
        # DPTReassembleLayer outputs decoder_channels channels, and the intermediate
        # conv is a simple identity-channel projection (decoder_channels → decoder_channels).
        C = self.decoder_channels
        dpt_cfg = DPTConfig(
            hidden_size=self.hidden_size,
            neck_hidden_sizes=[C, C, C, C],
            reassemble_factors=[4, 2, 1, 0.5],
            readout_type="ignore",
            fusion_hidden_size=C,
            is_hybrid=False,
            use_batch_norm_in_fusion_residual=False,
            use_bias_in_fusion_residual=True,
        )
        self.reassemble_stage = DPTReassembleStage(dpt_cfg)

        # Intermediate 3×3 conv projections: neck_hidden_sizes[i] → fusion_hidden_size.
        # Since both are decoder_channels, these are channel-preserving convs
        # (matching what DPTNeck.convs does).
        self.neck_convs = nn.ModuleList([
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
            for _ in range(4)
        ])

        self.fusion_stage = DPTFeatureFusionStage(dpt_cfg)

    def _reassemble(self, token_sets: list, Hp: int, Wp: int) -> list:
        """Reshape token sets into 2-D feature maps at strides 4, 8, 16, 32.

        Args:
            token_sets: list[Tensor(B, N_tokens, hidden_size)], len == len(tap_layers)
                        N_tokens = (Hp // patch_size) * (Wp // patch_size).
                        CLS token is already dropped by _encode.
            Hp, Wp: padded spatial dimensions (multiples of patch_size).

        Returns:
            list of 4 tensors:
              (B, decoder_channels, Hp/4,  Wp/4)
              (B, decoder_channels, Hp/8,  Wp/8)
              (B, decoder_channels, Hp/16, Wp/16)
              (B, decoder_channels, Hp/32, Wp/32)

        CLS-token handling: DPTReassembleStage.forward always splits off token
        index 0 as the CLS token (cls_token, hidden_state = hs[:, 0], hs[:, 1:]),
        regardless of readout_type. We prepend a dummy zero token so the split
        discards only that dummy; with readout_type="ignore" the dummy value is
        never used in any computation.
        """
        patch_h = Hp // self.patch_size
        patch_w = Wp // self.patch_size

        # Prepend a dummy CLS token (zeros) to each token set so DPTReassembleStage
        # can safely do its [:, 0] / [:, 1:] split without consuming real patch tokens.
        padded = []
        for ts in token_sets:
            B, _, C = ts.shape
            dummy_cls = torch.zeros(B, 1, C, device=ts.device, dtype=ts.dtype)
            padded.append(torch.cat([dummy_cls, ts], dim=1))  # (B, N+1, C)

        maps = self.reassemble_stage(padded, patch_height=patch_h, patch_width=patch_w)
        return maps

    def _fuse(self, maps: list) -> torch.Tensor:
        """Fuse the 4 multi-scale reassembled maps into a single feature map.

        Args:
            maps: list of 4 tensors as produced by _reassemble.
                  shapes: /4, /8, /16, /32 of the padded input.

        Returns:
            Tensor(B, decoder_channels, Hp/2, Wp/2)

        Output stride analysis: DPTFeatureFusionLayer.forward applies
        F.interpolate(scale_factor=2) at the end of each fusion step.
        The stage processes 4 inputs in reverse order (coarsest first: /32, /16,
        /8, /4). After 4 rounds of ×2 upsampling: /32 × 2^4 = /2. The final
        fused output is at stride /2, i.e. exactly (Hp/2, Wp/2).
        """
        # Project each reassembled map through the intermediate conv before fusion
        projected = [conv(m) for conv, m in zip(self.neck_convs, maps)]
        # DPTFeatureFusionStage.forward calls layers[0] with no residual, so
        # layers[0].residual_layer1 never runs and receives no gradient. Replicate
        # the coarse→fine fusion manually, seeding layer 0 with the coarsest map as
        # its own residual so every fusion parameter participates in the graph.
        layers = self.fusion_stage.layers
        rev = projected[::-1]  # coarsest (/32) first, finest (/4) last
        fused = layers[0](rev[0], rev[0])
        for feat, layer in zip(rev[1:], layers[1:]):
            # Align the coarse-derived map to the finer map's exact size BEFORE
            # merging. DPTFeatureFusionLayer otherwise resizes the finer (more
            # accurate) residual DOWN to the coarse-derived size; when a patch-grid
            # axis is odd the /32 reassemble conv rounds up (e.g. width 5 → 3, not
            # 2.5), so that coarse size is wrong and the finer maps get stretched in
            # that axis only. Resizing the coarse map instead keeps the finer maps
            # undistorted and makes the chain self-align to an exact (Hp/2, Wp/2).
            fused = F.interpolate(fused, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            fused = layer(fused, feat)
        # fused is the finest fused map, at exactly /2 of the padded input.
        return fused

    def _final(self, dense: torch.Tensor, Hp: int, Wp: int, H: int, W: int) -> torch.Tensor:
        """Upsample fused map to full padded resolution (Hp, Wp) then crop to (H, W).

        Args:
            dense: fused feature map at /2 of the padded input, i.e. (Hp/2, Wp/2)
                (guaranteed exact by _fuse's size-aligned fusion). We upsample to
                the padded size (Hp, Wp) passed from forward_one — a clean 2× in
                both axes — then crop, so geometry is preserved.
            Hp, Wp: padded spatial dimensions (multiples of patch_size).
            H, W: original (unpadded) spatial dimensions.

        Returns:
            Tensor(B, decoder_channels, H, W)
        """
        up = F.interpolate(dense, size=(Hp, Wp), mode='bilinear', align_corners=False)
        return self._crop(up, H, W)

    # ------------------------------------------------------------------ #
    # forward_one — full pad→encode→reassemble→fuse→final→proj→heads flow  #
    # ------------------------------------------------------------------ #

    def forward_one(self, x: torch.Tensor) -> dict:
        """Process a single image tensor (B, 3, H, W) through the full pipeline."""
        # 1. Pad to patch_size multiple
        x_pad, (H, W), _ = self._pad_to_multiple(x)
        _, _, Hp, Wp = x_pad.shape

        # 2. Encode to token sets at tap_layers depths
        token_sets = self._encode(x_pad)

        # 3. Reassemble tokens into multi-scale feature maps
        maps = self._reassemble(token_sets, Hp, Wp)

        # 4. Fuse maps → /2 feature map (decoder_channels)
        fused = self._fuse(maps)

        # 5. Upsample to /1 and crop to original resolution (decoder_channels)
        dense = self._final(fused, Hp, Wp, H, W)

        # 6. Project decoder_channels → out_dim
        dense = self.proj(dense)

        # 7. Heads on dense**2, matching Quad_L2Net_ConfCFS / ConvNeXtV2
        ureliability = self.clf(dense ** 2)
        urepeatability = self.sal(dense ** 2)

        result = self.normalize(dense, ureliability, urepeatability)

        # Hard resolution contract — NMS and extract_multiscale index at input resolution
        assert result['descriptors'].shape[-2:] == (H, W), (
            f"Resolution contract violated: descriptors {result['descriptors'].shape[-2:]} != ({H}, {W})"
        )

        return result
