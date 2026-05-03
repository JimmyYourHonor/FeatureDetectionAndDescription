# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import numpy as np
from PIL import Image
import random
from math import ceil

from . import transforms_tools as F


class Scale(object):
    """ Rescale the input PIL.Image to a given size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    The smallest dimension of the resulting image will be = size.

    if largest == True: same behaviour for the largest dimension.

    if not can_upscale: don't upscale
    if not can_downscale: don't downscale
    """
    def __init__(self, size, interpolation=Image.BILINEAR, largest=False,
                 can_upscale=True, can_downscale=True):
        assert isinstance(size, int) or (len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.largest = largest
        self.can_upscale = can_upscale
        self.can_downscale = can_downscale

    def __repr__(self):
        fmt_str = "RandomScale(%s" % str(self.size)
        if self.largest: fmt_str += ', largest=True'
        if not self.can_upscale: fmt_str += ', can_upscale=False'
        if not self.can_downscale: fmt_str += ', can_downscale=False'
        return fmt_str+')'

    def get_params(self, imsize):
        w,h = imsize
        if isinstance(self.size, int):
            cmp = lambda a,b: (a>=b) if self.largest else (a<=b)
            if (cmp(w, h) and w == self.size) or (cmp(h, w) and h == self.size):
                ow, oh = w, h
            elif cmp(w, h):
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
        else:
            ow, oh = self.size
        return ow, oh

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        size2 = ow, oh = self.get_params(img.size)

        if size2 != img.size:
            a1, a2 = img.size, size2
            if (self.can_upscale and min(a1) < min(a2)) or (self.can_downscale and min(a1) > min(a2)):
                img = img.resize(size2, self.interpolation)

        return F.update_img_and_labels(inp, img, persp=(ow/w,0,0,0,oh/h,0,0,0))


class RandomScale(Scale):
    """Rescale the input PIL.Image to a random size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        min_size (int): min size of the smaller edge of the picture.
        max_size (int): max size of the smaller edge of the picture.

        ar (float or tuple):
            max change of aspect ratio (width/height).

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min_size, max_size, ar=1,
                 can_upscale=False, can_downscale=True, interpolation=Image.BILINEAR):
        Scale.__init__(self, 0, can_upscale=can_upscale, can_downscale=can_downscale, interpolation=interpolation)
        assert type(min_size) == type(max_size), 'min_size and max_size can only be 2 ints or 2 floats'
        assert isinstance(min_size, int) and min_size >= 1 or isinstance(min_size, float) and min_size>0
        assert isinstance(max_size, (int,float)) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        if type(ar) in (float,int): ar = (min(1/ar,ar),max(1/ar,ar))
        assert 0.2 < ar[0] <= ar[1] < 5
        self.ar = ar

    def get_params(self, imsize):
        w,h = imsize
        if isinstance(self.min_size, float):
            min_size = int(self.min_size*min(w,h) + 0.5)
        if isinstance(self.max_size, float):
            max_size = int(self.max_size*min(w,h) + 0.5)
        if isinstance(self.min_size, int):
            min_size = self.min_size
        if isinstance(self.max_size, int):
            max_size = self.max_size

        if not self.can_upscale:
            max_size = min(max_size,min(w,h))

        size = int(0.5 + F.rand_log_uniform(min_size,max_size))
        ar = F.rand_log_uniform(*self.ar) # change of aspect ratio

        if w < h: # image is taller
            ow = size
            oh = int(0.5 + size * h / w / ar)
            if oh < min_size:
                ow,oh = int(0.5 + ow*float(min_size)/oh),min_size
        else: # image is wider
            oh = size
            ow = int(0.5 + size * w / h * ar)
            if ow < min_size:
                ow,oh = min_size,int(0.5 + oh*float(min_size)/ow)

        assert ow >= min_size, 'image too small (width=%d < min_size=%d)' % (ow, min_size)
        assert oh >= min_size, 'image too small (height=%d < min_size=%d)' % (oh, min_size)
        return ow, oh


class RandomTilting(object):
    """Apply a random tilting (left, right, up, down) to the input PIL.Image
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        maginitude (float):
            maximum magnitude of the random skew (value between 0 and 1)
        directions (string):
            tilting directions allowed (all, left, right, up, down)
            examples: "all", "left,right", "up-down-right"
    """

    def __init__(self, magnitude, directions='all'):
        self.magnitude = magnitude
        self.directions = directions.lower().replace(',',' ').replace('-',' ')

    def __repr__(self):
        return "RandomTilt(%g, '%s')" % (self.magnitude,self.directions)

    def get_params(self, src_size):
        """Sample tilt parameters and return the forward (input→output) homography.

        Args:
            src_size: (w, h) tuple — source image size in pixels.

        Returns:
            8-tuple representing the input→output perspective transform, in the
            same packing convention used by ``persp_mul``.
        """
        w, h = src_size

        x1, y1, x2, y2 = 0, 0, h, w
        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.directions == 'all':
            choices = [0, 1, 2, 3]
        else:
            dirs = ['left', 'right', 'up', 'down']
            choices = []
            for d in self.directions.split():
                try:
                    choices.append(dirs.index(d))
                except:
                    raise ValueError('Tilting direction %s not recognized' % d)

        skew_direction = random.choice(choices)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

        # PIL.Image.PERSPECTIVE expects an output→input homography; solve for it
        # via the standard 8-point system (see https://goo.gl/sSgJdj).
        matrix = []
        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(original_plane).reshape(8)

        pil_inverse = np.dot(np.linalg.pinv(A), B)
        pil_inverse = tuple(np.array(pil_inverse).reshape(8))

        forward = np.linalg.pinv(np.float32(pil_inverse + (1,)).reshape(3, 3)).ravel()[:8]
        return tuple(forward)

    def __call__(self, inp):
        img = F.grab_img(inp)
        forward = self.get_params(img.size)

        # PIL.Image.transform needs the output→input homography
        pil_inverse = np.linalg.pinv(
            np.float32(forward + (1,)).reshape(3, 3)
        ).ravel()[:8]
        img = img.transform(
            img.size, Image.PERSPECTIVE, tuple(pil_inverse), resample=Image.BICUBIC
        )
        return F.update_img_and_labels(inp, img, persp=forward)
