import numpy as np
import torch
from tools.transforms import *
from .gpu_window_select import MODE_ANALYTIC, MODE_FLOW


class ParametricTransform:
    """CPU-side transform that defers image rendering AND window selection
    to the GPU.

    Samples geometric transform parameters (RandomScale, RandomTilting) and
    builds the 3×3 chain matrices, then emits per sample:

      ``src_a`` / ``src_b``:
          Raw uint8 ``(3, H, W)`` source-image tensors (variable resolution,
          carried through as Python lists by the collator).
      ``sa2ia`` / ``sb2ib``:
          ``(3, 3)`` float32 src→img homographies.
      ``M_ab``:
          ``(3, 3)`` float32 img_a→img_b homography. Analytic for synthetic
          and still modes; identity placeholder in flow mode (where
          ``aflow_full`` provides the per-pixel mapping).
      ``img_size``:
          ``(4,)`` int [W_a, H_a, W_b, H_b].
      ``mode``:
          0 = analytic (synthetic/still), 1 = flow.
      ``aflow_full``:
          ``(2, H_a, W_a)`` float32 per-pixel img_a→img_b coords. Real flow
          for mode=1; ``(2, 1, 1)`` zero placeholder for mode=0.
      ``mask_full``:
          ``(H_a, W_a)`` uint8 validity mask. Real for mode=1; ``(1, 1)``
          all-ones placeholder for mode=0.

    The per-pixel ``aflow_full``/``mask_full`` carry through the variable
    resolution pipeline as Python lists; only the fixed-shape 3×3 matrices
    and ``img_size`` get stacked. Window selection, crop rendering, and
    augmentation all happen on GPU downstream.
    """

    def __init__(self,
                 synthetic_scale=RandomScale(256, 1024, can_upscale=True),
                 synthetic_tilt=RandomTilting(0.5),
                 still_scale=RandomScale(256, 1024, can_upscale=True),
                 still_tilt=RandomTilting(0.5),
                 second_scale=RandomScale(256, 1024, can_upscale=True)):
        self.synthetic_scale = synthetic_scale
        self.synthetic_tilt = synthetic_tilt
        self.still_scale = still_scale
        self.still_tilt = still_tilt
        self.second_scale = second_scale

    # ── coordinate helpers ────────────────────────────────────────────────────

    @staticmethod
    def _identity():
        return np.eye(3, dtype=np.float32)

    @staticmethod
    def _scale_h(src_size, dst_size):
        """3x3 'src pixel coord → dst pixel coord' for an axis-aligned resize."""
        Ws, Hs = src_size
        Wd, Hd = dst_size
        return np.array([[Wd / Ws, 0, 0],
                         [0, Hd / Hs, 0],
                         [0, 0, 1]], dtype=np.float32)

    @staticmethod
    def _tilt_h(homography_8tuple):
        return np.asarray(homography_8tuple + (1.0,), dtype=np.float32).reshape(3, 3)

    @staticmethod
    def _pil_to_uint8_tensor(pil_img):
        arr = np.array(pil_img.convert('RGB'))
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).contiguous()

    # ── per-mode setup ────────────────────────────────────────────────────────

    def _synthetic_setup(self, image):
        src = image.convert('RGB')
        src_tensor = self._pil_to_uint8_tensor(src)

        scale_a_size = self.synthetic_scale.get_params(src.size)
        scale_a = self._scale_h(src.size, scale_a_size)
        tilt = self._tilt_h(self.synthetic_tilt.get_params(scale_a_size))
        scale_b_size = self.second_scale.get_params(scale_a_size)
        scale_b = self._scale_h(scale_a_size, scale_b_size)

        sa2ia = scale_a
        sb2ib = scale_b @ tilt @ scale_a
        M_ab = scale_b @ tilt

        return src_tensor, src_tensor, sa2ia, sb2ib, M_ab, scale_a_size, scale_b_size

    def _still_setup(self, im0, im1):
        src_a = im0.convert('RGB')
        src_b = im1.convert('RGB')
        src_a_tensor = self._pil_to_uint8_tensor(src_a)
        src_b_tensor = self._pil_to_uint8_tensor(src_b)

        sa2ia = self._identity()
        img_a_size = src_a.size

        scale_b1_size = self.still_scale.get_params(src_b.size)
        scale_b1 = self._scale_h(src_b.size, scale_b1_size)
        tilt = self._tilt_h(self.still_tilt.get_params(scale_b1_size))
        scale_b2_size = self.second_scale.get_params(scale_b1_size)
        scale_b2 = self._scale_h(scale_b1_size, scale_b2_size)
        sb2ib = scale_b2 @ tilt @ scale_b1

        # img_a (= im0) → img_b: scale to im1 size, then through src_b chain.
        im0_to_im1 = self._scale_h(src_a.size, src_b.size)
        M_ab = sb2ib @ im0_to_im1

        return src_a_tensor, src_b_tensor, sa2ia, sb2ib, M_ab, img_a_size, scale_b2_size

    def _flow_setup(self, im0, im1, flow_pil, mask_pil):
        src_a = im0.convert('RGB')
        src_b = im1.convert('RGB')
        src_a_tensor = self._pil_to_uint8_tensor(src_a)
        src_b_tensor = self._pil_to_uint8_tensor(src_b)

        sa2ia = self._identity()
        img_a_size = src_a.size

        scale_b_size = self.second_scale.get_params(src_b.size)
        scale_b = self._scale_h(src_b.size, scale_b_size)
        sb2ib = scale_b
        M_ab = self._identity()  # placeholder; aflow_full carries the mapping

        # Decode per-pixel flow (RGBA bytes → 2×int16) and add the identity grid
        # to obtain im0→im1 correspondences, then compose with scale_b.
        W_a, H_a = img_a_size
        flow = np.array(flow_pil).view(np.int16).astype(np.float32) / 16.0  # (H_a, W_a, 2)
        ident = np.mgrid[:H_a, :W_a][::-1].transpose(1, 2, 0).astype(np.float32)
        aflow_im1 = flow + ident
        # Apply scale_b to land in img_b coords. Affine, so direct scale.
        sx = scale_b[0, 0]
        sy = scale_b[1, 1]
        aflow_full = np.stack([aflow_im1[..., 0] * sx, aflow_im1[..., 1] * sy], axis=0)
        mask_full = np.array(mask_pil, dtype=np.uint8)

        return (src_a_tensor, src_b_tensor, sa2ia, sb2ib, M_ab,
                img_a_size, scale_b_size, aflow_full, mask_full)

    # ── main ──────────────────────────────────────────────────────────────────

    def __call__(self, example):
        result = {
            'src_a': [], 'src_b': [],
            'sa2ia': [], 'sb2ib': [], 'M_ab': [],
            'img_size': [], 'mode': [],
            'aflow_full': [], 'mask_full': [],
        }
        for image, im0, im1, flow_pil, mask_pil in zip(
            example['image'], example['im0.jpg'], example['im1.jpg'],
            example['flow.png'], example['mask.png'],
        ):
            if image is not None:
                src_a, src_b, sa2ia, sb2ib, M_ab, img_a_size, img_b_size = \
                    self._synthetic_setup(image)
                aflow_full = np.zeros((2, 1, 1), dtype=np.float32)
                mask_full = np.ones((1, 1), dtype=np.uint8)
                mode = MODE_ANALYTIC
            elif im0 is not None and flow_pil is None:
                src_a, src_b, sa2ia, sb2ib, M_ab, img_a_size, img_b_size = \
                    self._still_setup(im0, im1)
                aflow_full = np.zeros((2, 1, 1), dtype=np.float32)
                mask_full = np.ones((1, 1), dtype=np.uint8)
                mode = MODE_ANALYTIC
            elif flow_pil is not None:
                (src_a, src_b, sa2ia, sb2ib, M_ab, img_a_size, img_b_size,
                 aflow_full, mask_full) = self._flow_setup(im0, im1, flow_pil, mask_pil)
                mode = MODE_FLOW
            else:
                raise ValueError("Invalid input data.")

            W_a, H_a = img_a_size
            W_b, H_b = img_b_size

            result['src_a'].append(src_a)
            result['src_b'].append(src_b)
            result['sa2ia'].append(torch.from_numpy(sa2ia.astype(np.float32)))
            result['sb2ib'].append(torch.from_numpy(sb2ib.astype(np.float32)))
            result['M_ab'].append(torch.from_numpy(M_ab.astype(np.float32)))
            result['img_size'].append(torch.tensor([W_a, H_a, W_b, H_b], dtype=torch.int32))
            result['mode'].append(mode)
            result['aflow_full'].append(torch.from_numpy(np.ascontiguousarray(aflow_full)))
            result['mask_full'].append(torch.from_numpy(np.ascontiguousarray(mask_full)))
        return result
