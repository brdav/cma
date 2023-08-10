import io
import os
import zipfile
from typing import Any, Callable, List, Optional, Union

import requests
import torch
from PIL import Image

URLS = {
    "pseudo_labels_train_ACDC_cma_segformer": "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/626144/pseudo_labels_train_ACDC_cma_segformer.zip",
    "pseudo_labels_train_ACDC_cma_deeplabv2": "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/626144/pseudo_labels_train_ACDC_cma_deeplabv2.zip",
}


class ACDC(torch.utils.data.Dataset):

    orig_dims = (1080, 1920)

    def __init__(
            self,
            root: str,
            stage: str = "train",
            condition: Union[List[str], str] = [
                "fog", "night", "rain", "snow"],
            load_keys: Union[List[str], str] = [
                "image_ref", "image", "semantic"],
            transforms: Optional[Callable] = None,
            predict_on: str = "test",
            load_pseudo_labels: bool = False,
            pseudo_label_dir: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.load_pseudo_labels = load_pseudo_labels
        if self.load_pseudo_labels:
            assert pseudo_label_dir is not None
            _dpath = os.path.join(os.path.dirname(self.root), 'pseudo_labels')
            os.makedirs(_dpath, exist_ok=True)
            if not os.path.exists(os.path.join(_dpath, pseudo_label_dir)):
                print('Downloading and extracting pseudo-labels...')
                r = requests.get(URLS[pseudo_label_dir])
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(_dpath)
            self.pseudo_label_dir = os.path.join(
                os.path.dirname(self.root), 'pseudo_labels', pseudo_label_dir)

        assert stage in ["train", "val", "test", "predict"]
        self.stage = stage

        # mapping from stage to splits
        if self.stage == 'train':
            self.split = 'train'
        elif self.stage == 'val':
            self.split = 'val'
        elif self.stage == 'test':
            self.split = 'val'  # test on val split
        elif self.stage == 'predict':
            self.split = predict_on

        self.condition = [condition] if isinstance(
            condition, str) else condition

        self.load_keys = [load_keys] if isinstance(
            load_keys, str) else load_keys

        self.paths = {k: []
                      for k in ['image', 'image_ref', 'semantic']}

        self.images_dir = os.path.join(self.root, 'rgb_anon')
        self.semantic_dir = os.path.join(self.root, 'gt')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.semantic_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "condition" are inside the "root" directory')

        for cond in self.condition:
            img_parent_dir = os.path.join(self.images_dir, cond, self.split)
            semantic_parent_dir = os.path.join(
                self.semantic_dir, cond, self.split)
            for recording in os.listdir(img_parent_dir):
                img_dir = os.path.join(img_parent_dir, recording)
                semantic_dir = os.path.join(semantic_parent_dir, recording)
                for file_name in os.listdir(img_dir):
                    for k in ['image', 'image_ref', 'semantic']:
                        if k == 'image':
                            file_path = os.path.join(img_dir, file_name)
                        elif k == 'image_ref':
                            ref_img_dir = img_dir.replace(
                                self.split, self.split + '_ref')
                            ref_file_name = file_name.replace(
                                'rgb_anon', 'rgb_ref_anon')
                            file_path = os.path.join(
                                ref_img_dir, ref_file_name)
                        elif k == 'semantic':
                            semantic_file_name = file_name.replace(
                                'rgb_anon.png', 'gt_labelTrainIds.png')
                            file_path = os.path.join(
                                semantic_dir, semantic_file_name)
                        self.paths[k].append(file_path)

    def __getitem__(self, index: int):

        sample: Any = {}
        if (not 'image' in self.load_keys) and ('image_ref' in self.load_keys):
            filename = self.paths['image_ref'][index].split('/')[-1]
        else:
            filename = self.paths['image'][index].split('/')[-1]
        sample['filename'] = filename

        for k in self.load_keys:
            if k in ['image', 'image_ref']:
                data = Image.open(self.paths[k][index]).convert('RGB')
            elif k == 'semantic':
                if self.load_pseudo_labels:
                    path = os.path.join(self.pseudo_label_dir, filename)
                    data = Image.open(path)
                else:
                    data = Image.open(self.paths[k][index])
            else:
                raise ValueError('invalid load_key')
            sample[k] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(next(iter(self.paths.values())))
