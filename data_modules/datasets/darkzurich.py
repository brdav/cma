import io
import os
import zipfile
from typing import Any, Callable, List, Optional, Union

import requests
import torch
from PIL import Image

URLS = {
    "pseudo_labels_train_DarkZurich_cma_segformer": "https://data.vision.ee.ethz.ch/brdavid/cma/pseudo_labels_train_DarkZurich_cma_segformer.zip",
}


class DarkZurich(torch.utils.data.Dataset):

    orig_dims = (1080, 1920)

    def __init__(
            self,
            root: str,
            stage: str = "train",
            load_keys: Union[List[str], str] = ["image_ref", "image"],
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

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        self.paths = {k: []
                      for k in ['image', 'image_ref', 'semantic']}

        self.images_dir = os.path.join(self.root, 'rgb_anon')
        self.semantic_dir = os.path.join(self.root, 'gt')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.semantic_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "condition" are inside the "root" directory')

        if self.split == 'train':
            img_ids = [i_id.strip() for i_id in open(os.path.join(
                os.path.dirname(__file__), 'lists/zurich_dn_pair_train.csv'))]
            for pair in img_ids:
                night, day = pair.split(",")
                for k in ['image', 'image_ref']:
                    if k == 'image':
                        file_path = os.path.join(
                            self.root, 'rgb_anon', night + "_rgb_anon.png")
                    elif k == 'image_ref':
                        file_path = os.path.join(
                            self.root, 'rgb_anon', day + "_rgb_anon.png")
                    self.paths[k].append(file_path)

        else:
            img_parent_dir = os.path.join(self.images_dir, self.split, 'night')
            semantic_parent_dir = os.path.join(
                self.semantic_dir, self.split, 'night')
            for recording in os.listdir(img_parent_dir):
                img_dir = os.path.join(img_parent_dir, recording)
                semantic_dir = os.path.join(semantic_parent_dir, recording)
                for file_name in os.listdir(img_dir):
                    for k in ['image', 'image_ref', 'semantic']:
                        if k == 'image':
                            file_path = os.path.join(img_dir, file_name)
                        elif k == 'image_ref':
                            if self.split == 'val':
                                ref_img_dir = img_dir.replace(self.split, self.split + '_ref').replace(
                                    'night', 'day').replace(recording, recording + '_ref')
                                ref_file_name = file_name.replace(
                                    'rgb_anon.png', 'ref_rgb_anon.png')
                                file_path = os.path.join(
                                    ref_img_dir, ref_file_name)
                            elif self.split == 'test':
                                ref_img_dir = img_dir.replace(self.split, self.split + '_ref').replace(
                                    'night', 'day').replace(recording, recording + '_ref')
                                ref_file_name_start = file_name.split('rgb_anon.png')[
                                    0]
                                for f in os.listdir(ref_img_dir):
                                    if f.startswith(ref_file_name_start):
                                        ref_file_name = f
                                        break
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
