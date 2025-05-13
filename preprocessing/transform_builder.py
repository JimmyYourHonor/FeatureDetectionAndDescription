import numpy as np
import torchvision.transforms as T
from tools.transforms import *
from tools.transforms_tools import persp_apply

class SyntheticPairTransformBuilder:
    """
    Builder class for creating synthetic pairs of images.
    This class is used to generate synthetic pairs of images for training.
    """

    def __init__(self,
                 scale=T.Compose([
                    RandomScale(256,1024,can_upscale=True)
                 ]),
                 distort=T.Compose([
                    RandomTilting(0.5), PixelNoise(25)
                 ])):
        self.scale = scale
        self.distort = distort

    def get_transform(self):
        def synthetic_pair_transform(example):
            """
            Create synthetic pairs of images for training.
            Applies scaling to the input image.
            Then applies distortion to create a second image.
            Finally, computes the affine flow between the two images.

            Args:
                example: The input image to be transformed.

            Returns:
                A dictionary containing the original and transformed images.
            """
            output_dict = {}
            scaled_img_list = [self.scale(img) for img in example['image']]
            output_dict['img_a'] = scaled_img_list
            distorted_scaled_img_list = [self.distort(
                dict(img=img, persp=(1,0,0,0,1,0,0,0))) for img in scaled_img_list]
            output_dict['img_b'] = [img['img'] for img in distorted_scaled_img_list]
            output_dict['aflow'] = []
            for entry in distorted_scaled_img_list:
                img = entry['img']
                trf = entry['persp']
                W, H = img.size
                xy = np.mgrid[0:H,0:W][::-1].reshape(2,H*W).T
                aflow = np.float32(persp_apply(trf, xy).reshape(H,W,2))
                output_dict['aflow'].append(aflow)
            return output_dict
            
        return synthetic_pair_transform
    
class StillPairTransformBuilder:
    """
    Builder class for creating pairs of identical images.
    This class is used to generate pairs where the second image is identical to the first.
    """

    def __init__(self,
                    transform=T.Compose([
                    RandomScale(256, 1024, can_upscale=True),
                    RandomTilting(0.5), PixelNoise(25)
                    ])):
        self.transform = transform

    def get_transform(self):
        def still_pair_transform(example):
            """
            Create identical pairs of images.
            Applies scaling only to img1 and uses img0 as is.
            Creates zero flow since images are identical.

            Args:
                example: Dictionary containing lists of img0 and img1.

            Returns:
                A dictionary containing the image pairs.
            """
            output_dict = {'img_a': [], 'img_b': [], 'aflow': []}

            for img0, img1 in zip(example['im0.jpg'], example['im1.jpg']):
                W, H = img0.size
                W2, H2 = img1.size
                sx = float(W2) / float(W)
                sy = float(H2) / float(H)
                mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1, 2, 0).astype(np.float32)
                aflow = mgrid * (sx, sy)

                transformed_img = self.transform(dict(img=img1, persp=(1, 0, 0, 0, 1, 0, 0, 0)))
                aflow_transformed = persp_apply(transformed_img['persp'], aflow.reshape(-1, 2)).reshape(aflow.shape)

                output_dict['img_a'].append(img0)
                output_dict['img_b'].append(transformed_img['img'])
                output_dict['aflow'].append(aflow_transformed)

            return output_dict

        return still_pair_transform
        
class FlowPairTransformBuilder:
    """
    Builder class for creating pairs of images with known optical flow.
    This class is used to transform image pairs that have ground truth flow between them.
    """

    def get_transform(self):
        def flow_pair_transform(example):
            """
            Transform image pairs with their corresponding flow.
            Applies transformation to both images and adjusts the flow accordingly.

            Args:
                example: Dictionary containing lists of img0, img1, and flow.

            Returns:
                A dictionary containing the transformed image pairs and adjusted flow.
            """
            output_dict = {'img_a': [], 'img_b': [], 'aflow': [], 'mask': []}

            for img0, img1, flow, mask in zip(example['im0.jpg'], example['im1.jpg'], example['flow.png'], example['mask.png']):
                W, H = img0.size
                flow = np.array(flow).view(np.int16)
                flow = np.float32(flow) / 16
                aflow = flow + np.mgrid[:H, :W][::-1].transpose(1, 2, 0)
                mask_array = np.array(mask)

                output_dict['img_a'].append(img0)
                output_dict['img_b'].append(img1)
                output_dict['aflow'].append(aflow)
                output_dict['mask'].append(mask_array)

            return output_dict

        return flow_pair_transform
