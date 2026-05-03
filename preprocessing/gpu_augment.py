"""Batched GPU augmentation: PixelNoise + ColorJitter + ImageNet normalize.

Runs on the uint8 (B, 3, H, W) crops emitted by GPUWarp. This module:

  * converts uint8 → float32 in [0, 1] on the GPU
  * applies pixel noise + color jitter to img_b only (training only)
  * normalizes both images with ImageNet mean/std

Color jitter is implemented in raw torch (no kornia / no torchvision functional
ops) and uses per-sample random factors. Hue shift uses an in-line RGB↔HSV
conversion.
"""

import torch
import torch.nn as nn


_LUMA = (0.299, 0.587, 0.114)  # ITU-R BT.601 Y' weights


class GPUBatchAugment(nn.Module):
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        noise_ampl: float = 25.0,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.noise_ampl = float(noise_ampl)

        self.register_buffer('_rgb_mean', torch.tensor(rgb_mean).view(1, 3, 1, 1))
        self.register_buffer('_rgb_std', torch.tensor(rgb_std).view(1, 3, 1, 1))
        self.register_buffer('_luma', torch.tensor(_LUMA).view(1, 3, 1, 1))

    def forward(self, img_a, img_b, training: bool = True):
        """Apply augmentation + normalization to a batched image pair.

        Args:
            img_a: (B, 3, H, W) uint8 or float
            img_b: (B, 3, H, W) uint8 or float
            training: if True, apply PixelNoise + ColorJitter to img_b.
                      If False, only normalize (used at eval time).
        """
        img_a = self._to_float01(img_a)
        img_b = self._to_float01(img_b)

        if training:
            img_b = self._pixel_noise(img_b)
            img_b = self._color_jitter(img_b)

        img_a = self._normalize(img_a)
        img_b = self._normalize(img_b)
        return img_a, img_b

    # ── low-level ops ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_float01(img):
        if img.dtype == torch.uint8:
            return img.float() * (1.0 / 255.0)
        return img.float()

    def _normalize(self, img):
        return (img - self._rgb_mean) / self._rgb_std

    def _pixel_noise(self, img):
        ampl = self.noise_ampl / 255.0
        noise = torch.empty_like(img).uniform_(-ampl / 2, ampl / 2)
        return (img + noise).clamp_(0.0, 1.0)

    def _color_jitter(self, img):
        B, _, _, _ = img.shape
        device = img.device

        if self.brightness > 0:
            f = torch.empty(B, 1, 1, 1, device=device).uniform_(
                max(0.0, 1.0 - self.brightness), 1.0 + self.brightness
            )
            img = (img * f).clamp_(0.0, 1.0)

        if self.contrast > 0:
            f = torch.empty(B, 1, 1, 1, device=device).uniform_(
                max(0.0, 1.0 - self.contrast), 1.0 + self.contrast
            )
            gray = (img * self._luma).sum(dim=1, keepdim=True)
            mean = gray.mean(dim=(2, 3), keepdim=True)
            img = ((img - mean) * f + mean).clamp_(0.0, 1.0)

        if self.saturation > 0:
            f = torch.empty(B, 1, 1, 1, device=device).uniform_(
                max(0.0, 1.0 - self.saturation), 1.0 + self.saturation
            )
            gray = (img * self._luma).sum(dim=1, keepdim=True)
            img = ((img - gray) * f + gray).clamp_(0.0, 1.0)

        if self.hue > 0:
            shift = torch.empty(B, 1, 1, 1, device=device).uniform_(-self.hue, self.hue)
            img = self._adjust_hue(img, shift)

        return img

    # ── hue adjustment via inline RGB↔HSV ─────────────────────────────────────

    def _adjust_hue(self, img, hue_shift):
        hsv = self._rgb_to_hsv(img)
        h = (hsv[:, 0:1] + hue_shift) % 1.0
        hsv = torch.cat([h, hsv[:, 1:2], hsv[:, 2:3]], dim=1)
        return self._hsv_to_rgb(hsv).clamp_(0.0, 1.0)

    @staticmethod
    def _rgb_to_hsv(rgb):
        # rgb: (B, 3, H, W) in [0, 1]
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]
        cmax, _ = rgb.max(dim=1, keepdim=True)
        cmin, _ = rgb.min(dim=1, keepdim=True)
        delta = cmax - cmin
        safe_delta = delta.clamp(min=1e-12)

        is_r = cmax == r
        is_g = (cmax == g) & ~is_r
        # implicit else: blue

        h_r = ((g - b) / safe_delta) % 6.0
        h_g = (b - r) / safe_delta + 2.0
        h_b = (r - g) / safe_delta + 4.0
        h = torch.where(is_r, h_r, torch.where(is_g, h_g, h_b))
        h = torch.where(delta == 0, torch.zeros_like(h), h) / 6.0  # → [0, 1)

        s = torch.where(cmax == 0, torch.zeros_like(cmax), delta / cmax.clamp(min=1e-12))
        v = cmax
        return torch.cat([h, s, v], dim=1)

    @staticmethod
    def _hsv_to_rgb(hsv):
        h = hsv[:, 0:1]
        s = hsv[:, 1:2]
        v = hsv[:, 2:3]

        h6 = h * 6.0
        i = (torch.floor(h6) % 6.0).long()
        f = h6 - torch.floor(h6)

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        r = torch.where(i == 0, v,
            torch.where(i == 1, q,
            torch.where(i == 2, p,
            torch.where(i == 3, p,
            torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t,
            torch.where(i == 1, v,
            torch.where(i == 2, v,
            torch.where(i == 3, q,
            torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p,
            torch.where(i == 1, p,
            torch.where(i == 2, t,
            torch.where(i == 3, v,
            torch.where(i == 4, v, q)))))
        return torch.cat([r, g, b], dim=1)
