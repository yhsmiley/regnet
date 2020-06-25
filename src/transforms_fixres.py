import torch
import numbers
import numpy as np
from PIL import Image
from src.config import *
from torchvision import transforms
import torchvision.transforms.functional as F


class Resize(transforms.Resize):
    """
    Resize with a ``largest=False'' argument
    allowing to resize to a common largest side without cropping
    """
    def __init__(self, size, largest=False, **kwargs):
        super().__init__(size, **kwargs)
        self.largest = largest

    @staticmethod
    def target_size(w, h, size, largest=False):
        if h < w and largest:
            w, h = size, int(size * h / w)
        else:
            w, h = int(size * w / h), size
        size = (h, w)
        return size

    def __call__(self, img):
        size = self.size
        w, h = img.size
        target_size = self.target_size(w, h, size, self.largest)
        return F.resize(img, target_size, self.interpolation)

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ', largest={})'.format(self.largest)


class CenterCrop(object):
    """Crops the given PIL Image at the center.
        Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is
        made.
        """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def _is_pil_image(img):
        return isinstance(img, Image.Image)

    def crop(self, img, i, j, h, w):
        """Crop the given PIL Image.
            Args:
            img (PIL Image): Image to be cropped.
            i (int): i in (i,j) i.e coordinates of the upper left corner.
            j (int): j in (i,j) i.e coordinates of the upper left corner.
            h (int): Height of the cropped image.
            w (int): Width of the cropped image.
            Returns:
            PIL Image: Cropped image.
            """
        if not self._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        
        return img.crop((j, i, j + w, i + h))

    def center_crop_new(self, img, output_size):
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        w, h = img.size
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        jit=0
        if j > 0:
            jit=np.random.randint(int(j+1))
        val=np.random.randint(2)
        scale=(1.0)*(val==0)+(-1.0)*(val==1)
        return self.crop(img, i, int(j+scale*jit), th, tw)

    def __call__(self, img):
        """
            Args:
            img (PIL Image): Image to be cropped.
            Returns:
            PIL Image: Cropped image.
            """
        return self.center_crop_new(img, self.size)
        
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Lighting(object):
    """
    PCA jitter transform on tensors
    """
    def __init__(self, alpha_std, eig_val, eig_vec):
        self.alpha_std = alpha_std
        self.eig_val = torch.as_tensor(eig_val, dtype=torch.float).view(1, 3)
        self.eig_vec = torch.as_tensor(eig_vec, dtype=torch.float)

    def __call__(self, data):
        if self.alpha_std == 0:
            return data
        alpha = torch.empty(1, 3).normal_(0, self.alpha_std)
        rgb = ((self.eig_vec * alpha) * self.eig_val).sum(1)
        data += rgb.view(3, 1, 1)
        data /= 1. + self.alpha_std
        return data


# FixRes
def get_transforms(kind='full', crop=True, need=('train', 'val')):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transformations = {}
    if 'train' in need:
        if kind == 'torch':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(TRAIN_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Lighting(0.1, np.array(EIGENVALUES), np.array(EIGENVECTORS)),
                transforms.Normalize(mean, std),
            ])
        elif kind == 'full':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(TRAIN_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.ToTensor(),
                Lighting(0.1, np.array(EIGENVALUES), np.array(EIGENVECTORS)),
                transforms.Normalize(mean, std),
            ])
        else:
            raise ValueError('Transforms kind {} unknown'.format(kind))
    if 'val' in need:
        if crop:
            transformations['val'] = transforms.Compose(
                [Resize(int((256 / 224) * TEST_IMAGE_SIZE)),  # to maintain same ratio w.r.t. 224 images
                 transforms.CenterCrop(TEST_IMAGE_SIZE),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            transformations['val_train'] = transforms.Compose(
                [Resize(int((256 / 224) * TEST_IMAGE_SIZE)),  # to maintain same ratio w.r.t. 224 images
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.05, 0.05, 0.05),
                 CenterCrop(TEST_IMAGE_SIZE),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            transformations['val'] = transforms.Compose(
                [Resize(TEST_IMAGE_SIZE, largest=True), 
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
    return transformations
