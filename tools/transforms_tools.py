# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import numpy as np


class DummyImg:
    ''' This class is a dummy image only defined by its size.
    '''
    def __init__(self, size):
        self.size = size

    def resize(self, size, *args, **kwargs):
        return DummyImg(size)

    def transform(self, size, *args, **kwargs):
        return DummyImg(size)


def grab_img(img_and_label):
    ''' Called to extract the image from an img_and_label input
    (a dictionary). Also compatible with old-style PIL images.
    '''
    if isinstance(img_and_label, dict):
        try:
            return img_and_label['img']
        except KeyError:
            return DummyImg(img_and_label['imsize'])
    else:
        return img_and_label


def update_img_and_labels(img_and_label, img, persp=None):
    ''' Called to update the img_and_label
    '''
    if isinstance(img_and_label, dict):
        img_and_label['img'] = img
        img_and_label['imsize'] = img.size

        if persp:
            if 'persp' not in img_and_label:
                img_and_label['persp'] = (1,0,0,0,1,0,0,0)
            img_and_label['persp'] = persp_mul(persp, img_and_label['persp'])

        return img_and_label

    else:
        return img


def rand_log_uniform(a, b):
    return np.exp(np.random.uniform(np.log(a),np.log(b)))


def persp_mul(mat, mat2):
    ''' homography (perspective) multiplication.
    mat: 8-tuple (homography transform)
    mat2: 8-tuple (homography transform) or 2-tuple (point)
    '''
    assert isinstance(mat, tuple)
    assert isinstance(mat2, tuple)

    mat = np.float32(mat+(1,)).reshape(3,3)
    mat2 = np.array(mat2+(1,)).reshape(3,3)
    res = np.dot(mat, mat2)
    return tuple((res/res[2,2]).ravel()[:8])
