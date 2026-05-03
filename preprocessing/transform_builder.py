import numpy as np
from PIL import Image
import torch
from tools.transforms import *

class ParametricTransform:
    """CPU-side transform that defers image rendering to the GPU.

    Samples geometric transform parameters (RandomScale, RandomTilting) and
    crop windows without ever rendering an image. Emits, per sample:

      ``src_a`` / ``src_b``:
          Raw uint8 ``(3, H, W)`` source-image tensors. For *synthetic* mode
          the same source feeds both branches.
      ``M_a`` / ``M_b``:
          ``(3, 3)`` float32 homographies that map *crop* pixel coords to the
          corresponding *source* pixel coords (pixel-corner convention, simple
          ``out/in`` scaling — same convention as ``persp_apply``).
      ``aflow``:
          ``(2, crop_H, crop_W)`` float32 — for crop_a pixel ``(j, i)``, the
          corresponding ``(x, y)`` coord in crop_b. NaN where invalid.
      ``mask``:
          ``(crop_H, crop_W)`` uint8 — non-zero where the flow is valid.

    Image rendering is performed downstream by ``GPUWarp`` via
    ``F.grid_sample`` on the batched stack.
    """

    def __init__(self,
                 synthetic_scale=RandomScale(256, 1024, can_upscale=True),
                 synthetic_tilt=RandomTilting(0.5),
                 still_scale=RandomScale(256, 1024, can_upscale=True),
                 still_tilt=RandomTilting(0.5),
                 second_scale=RandomScale(256, 1024, can_upscale=True),
                 crop_size=(192, 192)):
        self.synthetic_scale = synthetic_scale
        self.synthetic_tilt = synthetic_tilt
        self.still_scale = still_scale
        self.still_tilt = still_tilt
        self.second_scale = second_scale
        self.crop_size = crop_size
        self.n_samples = 3

    # ── coordinate helpers ────────────────────────────────────────────────────

    @staticmethod
    def _identity():
        return np.eye(3, dtype=np.float64)

    @staticmethod
    def _scale_h(src_size, dst_size):
        """3x3 'src pixel coord → dst pixel coord' for an axis-aligned resize.

        Uses simple ratio scaling (no ``-1`` offset), matching the
        ``persp_apply`` convention used elsewhere in this codebase.
        """
        Ws, Hs = src_size
        Wd, Hd = dst_size
        return np.array([[Wd / Ws, 0, 0],
                         [0, Hd / Hs, 0],
                         [0, 0, 1]], dtype=np.float64)

    @staticmethod
    def _tilt_h(homography_8tuple):
        """Convert ``RandomTilting.get_params`` 8-tuple → 3x3 ndarray."""
        h = np.asarray(homography_8tuple + (1.0,), dtype=np.float64).reshape(3, 3)
        return h

    @staticmethod
    def _apply_h(M, xy):
        """Apply a 3x3 homography to (N, 2) ``(x, y)`` coords."""
        n = xy.shape[0]
        homo = np.concatenate([xy, np.ones((n, 1), dtype=xy.dtype)], axis=1)
        out = homo @ M.T
        out[:, :2] /= out[:, 2:3]
        return out[:, :2]

    @staticmethod
    def _pil_to_uint8_tensor(pil_img):
        arr = np.array(pil_img.convert('RGB'))
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).contiguous()

    # ── per-mode setup: returns sources, src→img homographies, aflow, mask ────

    def _synthetic_setup(self, image):
        src = image.convert('RGB')
        scale_a_size = self.synthetic_scale.get_params(src.size)
        scale_a = self._scale_h(src.size, scale_a_size)

        tilt = self._tilt_h(self.synthetic_tilt.get_params(scale_a_size))
        # PIL.Image.PERSPECTIVE preserves the input size, so img_b_pre.size == img_a.size
        scale_b_size = self.second_scale.get_params(scale_a_size)
        scale_b = self._scale_h(scale_a_size, scale_b_size)

        src_to_img_a = scale_a
        src_to_img_b = scale_b @ tilt @ scale_a

        # aflow at img_a resolution: img_a pixel (x, y) → img_b coord
        img_a_to_img_b = scale_b @ tilt
        W_a, H_a = scale_a_size
        xy = np.mgrid[0:H_a, 0:W_a][::-1].reshape(2, -1).T.astype(np.float64)
        aflow = self._apply_h(img_a_to_img_b, xy).reshape(H_a, W_a, 2).astype(np.float32)
        mask = np.ones((H_a, W_a), dtype=np.uint8)

        return src, src, src_to_img_a, src_to_img_b, scale_a_size, scale_b_size, aflow, mask

    def _still_setup(self, im0, im1):
        src_a = im0.convert('RGB')
        src_b = im1.convert('RGB')

        # img_a == im0, no transform
        src_a_to_img_a = self._identity()
        img_a_size = src_a.size

        # img_b chain: scale_b1 → tilt → scale_b2
        scale_b1_size = self.still_scale.get_params(src_b.size)
        scale_b1 = self._scale_h(src_b.size, scale_b1_size)
        tilt = self._tilt_h(self.still_tilt.get_params(scale_b1_size))
        scale_b2_size = self.second_scale.get_params(scale_b1_size)
        scale_b2 = self._scale_h(scale_b1_size, scale_b2_size)
        src_b_to_img_b = scale_b2 @ tilt @ scale_b1

        # aflow at img_a resolution. The original pipeline assumes im0/im1 are
        # the same scene at different scales, so an im0 pixel at (x, y)
        # corresponds to im1 coord (x*W_im1/W_im0, y*H_im1/H_im0).
        im0_to_im1 = self._scale_h(src_a.size, src_b.size)
        # Compose with src_b_to_img_b to land in img_b coords
        im0_to_img_b = src_b_to_img_b @ im0_to_im1
        W_a, H_a = img_a_size
        xy = np.mgrid[0:H_a, 0:W_a][::-1].reshape(2, -1).T.astype(np.float64)
        aflow = self._apply_h(im0_to_img_b, xy).reshape(H_a, W_a, 2).astype(np.float32)
        mask = np.ones((H_a, W_a), dtype=np.uint8)

        return src_a, src_b, src_a_to_img_a, src_b_to_img_b, img_a_size, scale_b2_size, aflow, mask

    def _flow_setup(self, im0, im1, flow_pil, mask_pil):
        src_a = im0.convert('RGB')
        src_b = im1.convert('RGB')

        src_a_to_img_a = self._identity()
        img_a_size = src_a.size

        # img_b chain: just scale_b on im1
        scale_b_size = self.second_scale.get_params(src_b.size)
        scale_b = self._scale_h(src_b.size, scale_b_size)
        src_b_to_img_b = scale_b

        # Decode per-pixel flow (RGBA bytes → 2×int16) and add the identity grid
        # to obtain im0→im1 correspondences, then compose with scale_b.
        W_a, H_a = img_a_size
        flow = np.array(flow_pil).view(np.int16).astype(np.float32) / 16.0  # (H_a, W_a, 2)
        aflow_im1 = flow + np.mgrid[:H_a, :W_a][::-1].transpose(1, 2, 0).astype(np.float32)
        # Apply scale_b to land in img_b coords
        flat = aflow_im1.reshape(-1, 2).astype(np.float64)
        aflow = self._apply_h(scale_b, flat).reshape(H_a, W_a, 2).astype(np.float32)
        mask = np.array(mask_pil, dtype=np.uint8)

        return src_a, src_b, src_a_to_img_a, src_b_to_img_b, img_a_size, scale_b_size, aflow, mask

    # ── main ──────────────────────────────────────────────────────────────────

    def __call__(self, example):
        crop_W, crop_H = self.crop_size
        result = {
            'src_a': [], 'src_b': [],
            'M_a': [], 'M_b': [],
            'aflow': [], 'mask': [],
        }
        for image, im0, im1, flow_pil, mask_pil in zip(example['image'], example['im0.jpg'],
                                                       example['im1.jpg'], example['flow.png'],
                                                       example['mask.png']):
            if image is not None:
                src_a, src_b, sa2ia, sb2ib, img_a_size, img_b_size, aflow, mask = \
                    self._synthetic_setup(image)
            elif im0 is not None and flow_pil is None:
                src_a, src_b, sa2ia, sb2ib, img_a_size, img_b_size, aflow, mask = \
                    self._still_setup(im0, im1)
            elif flow_pil is not None:
                src_a, src_b, sa2ia, sb2ib, img_a_size, img_b_size, aflow, mask = \
                    self._flow_setup(im0, im1, flow_pil, mask_pil)
            else:
                raise ValueError("Invalid input data.")

            img_a_W, img_a_H = img_a_size
            img_b_W, img_b_H = img_b_size

            # ── crop-window selection ─────────────────────────────────────────
            dx = np.gradient(aflow[:, :, 0])
            dy = np.gradient(aflow[:, :, 1])
            scale = np.sqrt(np.clip(np.abs(dx[1] * dy[0] - dx[0] * dy[1]), 1e-16, 1e16))

            accu2 = np.zeros((16, 16), bool)
            Q = lambda x, w: np.int32(16 * (x - w.start) / (w.stop - w.start))

            def window1(x, size, w):
                l = x - int(0.5 + size / 2)
                r = l + int(0.5 + size)
                if l < 0: l, r = (0, r - l)
                if r > w: l, r = (l + w - r, w)
                if l < 0: l, r = 0, w
                return slice(l, r)

            def window(cx, cy, win_size, scale_factor, img_shape):
                return (window1(cy, win_size[1] * scale_factor, img_shape[0]),
                        window1(cx, win_size[0] * scale_factor, img_shape[1]))

            n_valid = mask.sum()
            sample_w = mask / (1e-16 + n_valid)

            def sample_valid_pixel():
                n = np.random.choice(sample_w.size, p=sample_w.ravel())
                y, x = np.unravel_index(n, sample_w.shape)
                return x, y

            trials = 0
            best = -np.inf, None
            for _ in range(30 * self.n_samples):
                if trials >= self.n_samples: break
                if n_valid == 0: break
                c1x, c1y = sample_valid_pixel()
                c2x, c2y = (aflow[c1y, c1x] + 0.5).astype(np.int32)
                if not (0 <= c2x < img_b_W and 0 <= c2y < img_b_H): continue
                sigma = scale[c1y, c1x]
                if 0.2 < sigma < 1:
                    win1 = window(c1x, c1y, self.crop_size, 1 / sigma, (img_a_H, img_a_W))
                    win2 = window(c2x, c2y, self.crop_size, 1, (img_b_H, img_b_W))
                elif 1 <= sigma < 5:
                    win1 = window(c1x, c1y, self.crop_size, 1, (img_a_H, img_a_W))
                    win2 = window(c2x, c2y, self.crop_size, sigma, (img_b_H, img_b_W))
                else:
                    continue
                x2, y2 = aflow[win1].reshape(-1, 2).T.astype(np.int32)
                valid = (win2[1].start <= x2) & (x2 < win2[1].stop) \
                      & (win2[0].start <= y2) & (y2 < win2[0].stop)
                score1 = (valid * mask[win1].ravel()).mean()
                accu2[:] = False
                accu2[Q(y2[valid], win2[0]), Q(x2[valid], win2[1])] = True
                score2 = accu2.mean()
                score = min(score1, score2)
                trials += 1
                if score > best[0]:
                    best = score, win1, win2

            if None in best:
                # Could not find a usable window — emit a degenerate sample.
                M_a = np.eye(3, dtype=np.float32)
                M_b = np.eye(3, dtype=np.float32)
                aflow_crop = np.full((2, crop_H, crop_W), np.nan, dtype=np.float32)
                mask_crop = np.zeros((crop_H, crop_W), dtype=np.uint8)
            else:
                win1, win2 = best[1:]
                ya, xa = win1
                yb, xb = win2
                W_w_a, H_w_a = xa.stop - xa.start, ya.stop - ya.start
                W_w_b, H_w_b = xb.stop - xb.start, yb.stop - yb.start

                # Win_a/Win_b: 'crop pixel coord → img pixel coord' homographies.
                Win_a = np.array([[W_w_a / crop_W, 0, xa.start],
                                  [0, H_w_a / crop_H, ya.start],
                                  [0, 0, 1]], dtype=np.float64)
                Win_b = np.array([[W_w_b / crop_W, 0, xb.start],
                                  [0, H_w_b / crop_H, yb.start],
                                  [0, 0, 1]], dtype=np.float64)
                M_a = (np.linalg.inv(sa2ia) @ Win_a).astype(np.float32)
                M_b = (np.linalg.inv(sb2ib) @ Win_b).astype(np.float32)

                # aflow_crop: per crop_a pixel index, the (x, y) coord in crop_b.
                # crop_a pixel-index (j, i) → img_a coord via Win_a (corner conv);
                # then through aflow (img_a → img_b) → through Win_b^-1 (img_b → crop_b).
                # We compute analytically for the synthetic/still cases and via
                # bilinear sampling of the per-pixel `aflow` for flow mode.
                # For uniformity we always slice + rescale `aflow`.
                aflow_w = aflow[win1] - np.float32([[[xb.start, yb.start]]])
                mask_w = mask[win1]
                aflow_w[~mask_w.view(bool)] = np.nan

                if (W_w_a, H_w_a) != self.crop_size:
                    afx = Image.fromarray(aflow_w[..., 0]).resize(self.crop_size, Image.NEAREST)
                    afy = Image.fromarray(aflow_w[..., 1]).resize(self.crop_size, Image.NEAREST)
                    aflow_crop = np.stack([np.float32(afx), np.float32(afy)])
                    mask_crop = np.asarray(
                        Image.fromarray(mask_w).resize(self.crop_size, Image.NEAREST)
                    )
                else:
                    aflow_crop = aflow_w.transpose(2, 0, 1)
                    mask_crop = mask_w

                if (W_w_b, H_w_b) != self.crop_size:
                    sx = crop_W / W_w_b
                    sy = crop_H / H_w_b
                    aflow_crop = aflow_crop * np.float32([[[sx]], [[sy]]])

            result['src_a'].append(self._pil_to_uint8_tensor(src_a))
            result['src_b'].append(self._pil_to_uint8_tensor(src_b))
            result['M_a'].append(torch.from_numpy(M_a))
            result['M_b'].append(torch.from_numpy(M_b))
            result['aflow'].append(torch.from_numpy(np.ascontiguousarray(aflow_crop)))
            result['mask'].append(torch.from_numpy(np.ascontiguousarray(mask_crop)))
        return result
