import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tools.transforms import *
from tools.transforms_tools import persp_apply

class SyntheticPairTransform:
    """
    A transform that creates a synthetic image from a real image.
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

    def __call__(self, image):
        """
        Create synthetic pairs of images for training.
        Applies scaling to the input image.
        Then applies distortion to create a second synthetic image.
        Finally, computes the affine flow between the two images.

        Args:
            example: The input image to be transformed.

        Returns:
            A dictionary containing the original and transformed images and absolute flow.
        """
        output_dict = {}
        output_dict['img_a'] = self.scale(image)
        distorted_scaled_image = self.distort(dict(img=output_dict['img_a'], persp=(1,0,0,0,1,0,0,0)))
        output_dict['img_b'] = distorted_scaled_image['img']
        W, H = output_dict['img_a'].size
        xy = np.mgrid[0:H,0:W][::-1].reshape(2,H*W).T
        output_dict['aflow'] = np.float32(persp_apply(distorted_scaled_image['persp'], xy).reshape(H,W,2))
        return output_dict
    
class StillPairTransform:
    """
    Transform class for transforming still image pairs with different lighting.
    The class applies random scaling, tilting, and pixel noise to the second 
    image of the pairs.
    """

    def __init__(self,
                    transform=T.Compose([
                    RandomScale(256, 1024, can_upscale=True),
                    RandomTilting(0.5), PixelNoise(25)
                    ])):
        self.transform = transform

    def __call__(self, im0, im1):
        """
        Transform pairs of images.
        Applies scaling only to img1 and uses img0 as is.
        Creates flow based on the scale difference between 
        the two images and the transformation applied to img1.
        Args:
            im0: First image in the pair.
            im1: Second image in the pair.
        Returns:
            A dictionary containing the new image pair and aflow.
        """
        output_dict = {}

        W, H = im0.size
        W2, H2 = im1.size
        sx = float(W2) / float(W)
        sy = float(H2) / float(H)
        mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1, 2, 0)
        aflow = mgrid * (sx, sy)

        transformed_img = self.transform(dict(img=im1, persp=(1, 0, 0, 0, 1, 0, 0, 0)))
        aflow_transformed = np.float32(persp_apply(transformed_img['persp'], aflow.reshape(-1, 2)).reshape(aflow.shape))

        output_dict['img_a'] = im0
        output_dict['img_b'] = transformed_img['img']
        output_dict['aflow'] = aflow_transformed

        return output_dict
        
class FlowPairTransform:
    def __call__(self, im0, im1, flow, mask):
        """
        Transform image pairs with their corresponding flow.
        Applies transformation to both images and adjusts the flow accordingly.

        Args:
            example: Dictionary containing lists of img0, img1, flow, and mask.

        Returns:
            A dictionary containing the transformed image pairs and adjusted flow.
        """
        output_dict = {}
        W, H = im0.size
        flow = np.array(flow).view(np.int16)
        flow = np.float32(flow) / 16
        aflow = flow + np.mgrid[:H, :W][::-1].transpose(1, 2, 0).astype(np.float32)
        mask_array = np.array(mask)

        output_dict['img_a'] = im0
        output_dict['img_b'] = im1
        output_dict['aflow'] = aflow
        output_dict['mask'] = mask_array

        return output_dict

class FullTransform:
    """
    Full transformation class for image pairs.
    This class applies different transformations to synthetic, still, and flow image pairs.
    It includes scaling, distortion, and cropping of the images.
    """

    def __init__(self, synthetic_scale=T.Compose([
                    RandomScale(256,1024,can_upscale=True)
                 ]),
                 synthetic_distort=T.Compose([
                    RandomTilting(0.5), PixelNoise(25)
                 ]),
                 still_transform=T.Compose([
                    RandomScale(256, 1024, can_upscale=True),
                    RandomTilting(0.5), PixelNoise(25)
                 ]),
                 scale=T.Compose([
                    RandomScale(256, 1024, can_upscale=True)
                 ]),
                 distort=T.Compose([
                    ColorJitter(0.2,0.2,0.2,0.1)
                 ]),
                 crop_size=(192, 192)):
        self.scale = scale
        self.distort = distort
        self.crop_size = crop_size
        self.n_samples = 5
        RGB_mean = [0.485, 0.456, 0.406]
        RGB_std  = [0.229, 0.224, 0.225]

        self.norm = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])
        self.synthetic_transform = SyntheticPairTransform(synthetic_scale, synthetic_distort)
        self.still_transform = StillPairTransform(still_transform)
        self.flow_transform = FlowPairTransform()

    def __call__(self, example):
        """
        Full transformation for the image pairs.
        Applies different transformations based on the type of image pair.
        It handles single image, still pairs, and flow pairs.
        Then applies the same kind of transformation for the unified formats.
        Finally, it crops the images to the specified size.
        Args:
            example: Dictionary containing lists of img_a, img_b,
                    flow, and mask.
        Returns:
            A dictionary containing the transformed image pairs,
            masks and aflows.
        """
        result = {'img_a': [], 'img_b': [], 'aflow': [], 'mask': []}
        for image, im0, im1, flow, mask in zip(example['image'], example['im0.jpg'], \
                                               example['im1.jpg'], example['flow.png'], \
                                               example['mask.png']):
            if image is not None:
                # Synthetic pair
                output_dict = self.synthetic_transform(image)
            elif im0 is not None and flow is None:
                # Still pair
                output_dict = self.still_transform(im0, im1)
            elif flow is not None:
                # Flow pair
                output_dict = self.flow_transform(im0, im1, flow, mask)
            else:
                raise ValueError("Invalid input data.")
            # Apply scaling and distortion to img_b
            img_dict = dict(img=output_dict['img_b'], persp=(1,0,0,0,1,0,0,0))
            img_dict = self.scale(img_dict)
            img_dict = self.distort(img_dict)
            img_b = img_dict['img']
            aflow = output_dict['aflow']
            # Apply the same transformation to the flow
            aflow = persp_apply(img_dict['persp'], aflow.reshape(-1,2)).reshape(aflow.shape)
            img_a = output_dict['img_a']
            mask = output_dict['mask'] if 'mask' in output_dict else np.ones(aflow.shape[:2],np.uint8)
            # Crop the images
            output_size_a = min(img_a.size, self.crop_size)
            output_size_b = min(img_b.size, self.crop_size)
            img_a = np.array(img_a)
            img_b = np.array(img_b)

            ah,aw,p1 = img_a.shape
            bh,bw,p2 = img_b.shape
            assert p1 == 3
            assert p2 == 3
            assert aflow.shape == (ah, aw, 2)
            assert mask.shape == (ah, aw)

            # Let's start by computing the scale of the
            # optical flow and applying a median filter:
            dx = np.gradient(aflow[:,:,0])
            dy = np.gradient(aflow[:,:,1])
            scale = np.sqrt(np.clip(np.abs(dx[1]*dy[0] - dx[0]*dy[1]), 1e-16, 1e16))

            accu2 = np.zeros((16,16), bool)
            Q = lambda x, w: np.int32(16 * (x - w.start) / (w.stop - w.start))
            
            def window1(x, size, w):
                l = x - int(0.5 + size / 2)
                r = l + int(0.5 + size)
                if l < 0: l,r = (0, r - l)
                if r > w: l,r = (l + w - r, w)
                if l < 0: l,r = 0,w # larger than width
                return slice(l,r)
            def window(cx, cy, win_size, scale, img_shape):
                return (window1(cy, win_size[1]*scale, img_shape[0]), 
                        window1(cx, win_size[0]*scale, img_shape[1]))

            n_valid_pixel = mask.sum()
            sample_w = mask / (1e-16 + n_valid_pixel)
            def sample_valid_pixel():
                n = np.random.choice(sample_w.size, p=sample_w.ravel())
                y, x = np.unravel_index(n, sample_w.shape)
                return x, y
            
            # Find suitable left and right windows
            trials = 0 # take the best out of few trials
            best = -np.inf, None
            for _ in range(50*self.n_samples):
                if trials >= self.n_samples: break # finished!

                # pick a random valid point from the first image
                if n_valid_pixel == 0: break
                c1x, c1y = sample_valid_pixel()
                
                # Find in which position the center of the left
                # window ended up being placed in the right image
                c2x, c2y = (aflow[c1y, c1x] + 0.5).astype(np.int32)
                if not(0 <= c2x < bw and 0 <= c2y < bh): continue

                # Get the flow scale
                sigma = scale[c1y, c1x]

                # Determine sampling windows
                if 0.2 < sigma < 1: 
                    win1 = window(c1x, c1y, output_size_a, 1/sigma, img_a.shape)
                    win2 = window(c2x, c2y, output_size_b, 1, img_b.shape)
                elif 1 <= sigma < 5:
                    win1 = window(c1x, c1y, output_size_a, 1, img_a.shape)
                    win2 = window(c2x, c2y, output_size_b, sigma, img_b.shape)
                else:
                    continue # bad scale

                # compute a score based on the flow
                x2,y2 = aflow[win1].reshape(-1, 2).T.astype(np.int32)
                # Check the proportion of valid flow vectors
                valid = (win2[1].start <= x2) & (x2 < win2[1].stop) \
                    & (win2[0].start <= y2) & (y2 < win2[0].stop)
                score1 = (valid * mask[win1].ravel()).mean()
                # check the coverage of the second window
                accu2[:] = False
                accu2[Q(y2[valid],win2[0]), Q(x2[valid],win2[1])] = True
                score2 = accu2.mean()
                # Check how many hits we got
                score = min(score1, score2)

                trials += 1
                if score > best[0]:
                    best = score, win1, win2
            
            if None in best: # counldn't find a good window
                img_a = np.zeros(output_size_a[::-1]+(3,), dtype=np.uint8)
                img_b = np.zeros(output_size_b[::-1]+(3,), dtype=np.uint8)
                aflow = np.nan * np.ones((2,)+output_size_a[::-1], dtype=np.float32)

            else:
                win1, win2 = best[1:]
                img_a = img_a[win1]
                img_b = img_b[win2]
                aflow = aflow[win1] - np.float32([[[win2[1].start, win2[0].start]]])
                mask = mask[win1]
                aflow[~mask.view(bool)] = np.nan # mask bad pixels!
                aflow = aflow.transpose(2,0,1) # --> (2,H,W)
                
                # rescale if necessary
                if img_a.shape[:2][::-1] != output_size_a:
                    sx, sy = (np.float32(output_size_a)-1)/(np.float32(img_a.shape[:2][::-1])-1)
                    img_a = np.asarray(Image.fromarray(img_a).resize(output_size_a, Image.LANCZOS))
                    mask = np.asarray(Image.fromarray(mask).resize(output_size_a, Image.NEAREST))
                    afx = Image.fromarray(aflow[0]).resize(output_size_a, Image.NEAREST)
                    afy = Image.fromarray(aflow[1]).resize(output_size_a, Image.NEAREST)
                    aflow = np.stack((np.float32(afx), np.float32(afy)))

                if img_b.shape[:2][::-1] != output_size_b:
                    sx, sy = (np.float32(output_size_b)-1)/(np.float32(img_b.shape[:2][::-1])-1)
                    img_b = np.asarray(Image.fromarray(img_b).resize(output_size_b, Image.LANCZOS))
                    aflow *= [[[sx]], [[sy]]]
            # Update the output dictionary
            result['img_a'].append(self.norm(img_a))
            result['img_b'].append(self.norm(img_b))
            assert aflow.dtype == np.float32
            result['aflow'].append(torch.tensor(aflow))
            result['mask'].append(torch.tensor(mask))
        return result