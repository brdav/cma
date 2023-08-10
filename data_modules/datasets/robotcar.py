import io
import os
import zipfile
from typing import Any, Callable, List, Optional, Union

import h5py
import numpy as np
import requests
import torch
from PIL import Image

URLS = {
    "pseudo_labels_train_RobotCar_cma_segformer": "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/626144/pseudo_labels_train_RobotCar_cma_segformer.zip",
}


class RobotCar(torch.utils.data.Dataset):

    ignore_index = 255
    id_to_trainid = {-1: ignore_index, 0: ignore_index, 1: ignore_index, 2: ignore_index,
                     3: ignore_index, 4: ignore_index, 5: ignore_index, 6: ignore_index,
                     7: 0, 8: 1, 9: ignore_index, 10: ignore_index, 11: 2, 12: 3, 13: 4,
                     14: ignore_index, 15: ignore_index, 16: ignore_index, 17: 5,
                     18: ignore_index, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                     28: 15, 29: ignore_index, 30: ignore_index, 31: 16, 32: 17, 33: 18}
    orig_dims = (1024, 1024)

    def __init__(
            self,
            root: str,
            stage: str = "train",
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
            self.split = 'test'
        elif self.stage == 'predict':
            self.split = predict_on

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" are inside the "root" directory')

        if self.split == 'train':
            self.images_dir = os.path.join(self.root, 'images')
            corr_dir = os.path.join(self.root, 'correspondence_data')
            self.paths = {'corr_files': []}
            f_name_list = [fn for fn in os.listdir(
                corr_dir) if fn.endswith('mat')]
            for f_name in f_name_list:
                self.paths['corr_files'].append(
                    os.path.join(corr_dir, f_name))
        else:
            assert not "image_ref" in self.load_keys
            self.paths = {k: [] for k in self.load_keys}
            splitdir_map = {'val': 'validation', 'test': 'testing'}
            images_dir = os.path.join(
                self.root, 'segmented_images', splitdir_map[self.split], 'imgs')
            semantic_dir = os.path.join(
                self.root, 'segmented_images', splitdir_map[self.split], 'annos')
            for img_name in os.listdir(images_dir):
                for k in self.load_keys:
                    if k == 'image':
                        file_path = os.path.join(images_dir, img_name)
                    elif k == 'semantic':
                        file_path = os.path.join(semantic_dir, img_name)
                    self.paths[k].append(file_path)

    def __getitem__(self, index: int):

        sample: Any = {}

        if self.split == 'train':
            # Load correspondence data
            mat_content = {}
            f = h5py.File(self.paths['corr_files'][index], 'r')
            for k, v in f.items():
                mat_content[k] = np.array(v)

            im1name = ''.join(chr(a[0])
                              for a in mat_content['im_i_path'])  # convert to string
            im2name = ''.join(chr(a[0])
                              for a in mat_content['im_j_path'])  # convert to string

            filename = im2name.split('/')[-1]
            sample['filename'] = filename

            for k in self.load_keys:
                if k == 'image_ref':
                    data = Image.open(os.path.join(
                        self.images_dir, im1name)).convert('RGB')
                elif k == 'image':
                    data = Image.open(os.path.join(
                        self.images_dir, im2name)).convert('RGB')
                elif k == 'semantic':
                    assert self.load_pseudo_labels
                    name = '.'.join([*filename.split('.')[:-1], 'png'])
                    path = os.path.join(self.pseudo_label_dir, name)
                    data = Image.open(path)
                sample[k] = data

        else:
            sample['filename'] = self.paths['image'][index].split('/')[-1]
            for k in self.load_keys:
                if k == 'image':
                    data = Image.open(self.paths[k][index]).convert('RGB')
                elif k == 'semantic':
                    data = Image.open(self.paths[k][index])
                    data = self.encode_semantic_map(data)
                sample[k] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(next(iter(self.paths.values())))

    @classmethod
    def encode_semantic_map(cls, semseg):
        semseg_arr = np.array(semseg)
        semseg_arr_copy = semseg_arr.copy()
        for k, v in cls.id_to_trainid.items():
            semseg_arr_copy[semseg_arr == k] = v
        return Image.fromarray(semseg_arr_copy.astype(np.uint8))
