import random

import torch
import torch.nn as nn
import torchvision

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


class ToTensor:
    def __init__(self, apply_keys='all'):
        self.apply_keys = apply_keys

    def __call__(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        elif self.apply_keys == 'none':
            apply_keys = []
        else:
            apply_keys = self.apply_keys

        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref']:
                sample[key] = torchvision.transforms.functional.pil_to_tensor(
                    val)
            elif key in ['semantic']:
                sample[key] = torchvision.transforms.functional.pil_to_tensor(
                    val).squeeze(0)
            elif key in ['filename']:
                pass
            else:
                raise ValueError

        return sample


class RandomCrop(nn.Module):
    def __init__(self, apply_keys='all', size=None, ignore_index=255, cat_max_ratio=1.0):
        super().__init__()
        self.apply_keys = apply_keys
        self.size = size
        self.ignore_index = ignore_index
        self.cat_max_ratio = cat_max_ratio

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        for k in ['image', 'image_ref', 'semantic']:
            if k in sample.keys():
                h, w = sample[k].shape[-2:]
        crop_params = self.get_params([h, w], self.size)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_tmp = self.crop(sample['semantic'], *crop_params)
                labels, cnt = torch.unique(seg_tmp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and cnt.max() / torch.sum(cnt).float() < self.cat_max_ratio:
                    break
                crop_params = self.get_params([h, w], self.size)
        for key in apply_keys:
            val = sample[key]
            if key in ['image',
                       'image_ref',
                       'semantic']:
                sample[key] = self.crop(val, *crop_params)
            elif key in ['filename']:
                pass
            else:
                raise ValueError('unknown key: {}'.format(key))
        return sample

    @staticmethod
    def get_params(img_size, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img_size
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, max(h - th, 0))
        j = random.randint(0, max(w - tw, 0))
        return i, j, min(th, h), min(tw, w)

    def crop(self, img, top, left, height, width):
        h, w = img.shape[-2:]
        right = left + width
        bottom = top + height

        if left < 0 or top < 0 or right > w or bottom > h:
            raise ValueError("Invalid crop parameters: {}, img size: {}".format(
                (top, left, height, width), (h, w)))
        return img[..., top:bottom, left:right]


class RandomHorizontalFlip(nn.Module):
    def __init__(self, apply_keys='all', p=0.5):
        super().__init__()
        self.apply_keys = apply_keys
        self.p = p

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        if random.random() < self.p:
            for key in apply_keys:
                val = sample[key]
                if key in ['image',
                           'image_ref',
                           'semantic']:
                    sample[key] = torchvision.transforms.functional.hflip(val)
                elif key in ['filename']:
                    pass
                else:
                    raise ValueError

        return sample


class ConvertImageDtype(torchvision.transforms.ConvertImageDtype):
    def __init__(self, apply_keys='all', **kwargs):
        dtype = kwargs.pop('dtype', torch.float)
        super().__init__(dtype=dtype, **kwargs)
        self.apply_keys = apply_keys

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref']:
                sample[key] = super().forward(val)
            elif key in ['semantic']:
                sample[key] = val.to(torch.long)  # from byte to long
            elif key in ['filename']:
                pass
            else:
                raise ValueError
        return sample


class Normalize(torchvision.transforms.Normalize):
    def __init__(self, apply_keys='all', **kwargs):
        # set imagenet statistics as default
        mean = kwargs.pop('mean', IMNET_MEAN)
        std = kwargs.pop('std', IMNET_STD)
        super().__init__(mean=mean, std=std, **kwargs)
        self.apply_keys = apply_keys

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref']:
                sample[key] = super().forward(val)
            elif key in ['semantic', 'filename']:
                pass
            else:
                raise ValueError
        return sample
