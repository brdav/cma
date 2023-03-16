import os
from itertools import chain
from operator import itemgetter
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.cli import instantiate_class
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from . import transforms as transform_lib
from .datasets import *

DATA_DIR = os.environ['DATA_DIR']


class CombinedDataModule(LightningDataModule):

    def __init__(
        self,
        load_config: dict,
        num_workers: int = 0,
        batch_size: int = 8,
        batch_size_divisor: int = 1,
        generate_pseudo_labels: bool = False,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dirs = {
            'ACDC': os.path.join(DATA_DIR, 'ACDC'),
            'DarkZurich': os.path.join(DATA_DIR, 'DarkZurich'),
            'RobotCar': os.path.join(DATA_DIR, 'RobotCar'),
            'ACG': DATA_DIR,
        }
        self.num_workers = num_workers
        assert batch_size % batch_size_divisor == 0
        self.batch_size_divisor = batch_size_divisor
        self.batch_size = batch_size // batch_size_divisor
        self.pin_memory = pin_memory
        self.generate_pseudo_labels = generate_pseudo_labels

        self.train_on = []
        self.train_config = []
        self.val_on = []
        self.val_config = []
        self.test_on = []
        self.test_config = []
        self.predict_on = []
        self.predict_config = []

        # parse load_config
        if 'train' in load_config:
            for ds, conf in load_config['train'].items():
                if isinstance(conf, dict):
                    self.train_on.append(ds)
                    self.train_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        self.train_on.append(ds)
                        self.train_config.append(el)

        if 'val' in load_config:
            for ds, conf in load_config['val'].items():
                if isinstance(conf, dict):
                    self.val_on.append(ds)
                    self.val_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        self.val_on.append(ds)
                        self.val_config.append(el)

        if 'test' in load_config:
            for ds, conf in load_config['test'].items():
                if isinstance(conf, dict):
                    self.test_on.append(ds)
                    self.test_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        self.test_on.append(ds)
                        self.test_config.append(el)

        if 'predict' in load_config:
            for ds, conf in load_config['predict'].items():
                if isinstance(conf, dict):
                    if self.generate_pseudo_labels:
                        conf['predict_on'] = 'train'
                    self.predict_on.append(ds)
                    self.predict_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        if self.generate_pseudo_labels:
                            el['predict_on'] = 'train'
                        self.predict_on.append(ds)
                        self.predict_config.append(el)

        self.idx_to_name = {'train': {}, 'val': {}, 'test': {}, 'predict': {}}
        for idx, ds in enumerate(self.train_on):
            self.idx_to_name['train'][idx] = ds
        for idx, ds in enumerate(self.val_on):
            self.idx_to_name['val'][idx] = ds
        for idx, ds in enumerate(self.test_on):
            self.idx_to_name['test'][idx] = ds
        for idx, ds in enumerate(self.predict_on):
            self.idx_to_name['predict'][idx] = ds

        if len(self.train_on) > 0:
            assert self.batch_size % len(
                self.train_on) == 0, 'batch size should be divisible by number of train datasets'

        # handle transformations
        for idx, (ds, cfg) in enumerate(zip(self.train_on, self.train_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.train_config[idx]['transforms'] = Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.train_config[idx]['transforms'] = transform_lib.ToTensor()
        for idx, (ds, cfg) in enumerate(zip(self.val_on, self.val_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.val_config[idx]['transforms'] = Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.val_config[idx]['transforms'] = transform_lib.ToTensor()
        for idx, (ds, cfg) in enumerate(zip(self.test_on, self.test_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.test_config[idx]['transforms'] = Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.test_config[idx]['transforms'] = transform_lib.ToTensor()
        for idx, (ds, cfg) in enumerate(zip(self.predict_on, self.predict_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.predict_config[idx]['transforms'] = Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.predict_config[idx]['transforms'] = transform_lib.ToTensor(
                )

        self.val_batch_size = 1
        self.test_batch_size = 1

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = []
            for ds, cfg in zip(self.train_on, self.train_config):
                self.train_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="train",
                    **cfg
                ))

        if stage in (None, "fit", "validate"):
            self.val_ds = []
            for ds, cfg in zip(self.val_on, self.val_config):
                self.val_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="val",
                    **cfg
                ))

        if stage in (None, "test"):
            self.test_ds = []
            for ds, cfg in zip(self.test_on, self.test_config):
                self.test_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="test",
                    **cfg
                ))

        if stage in (None, "predict"):
            self.predict_ds = []
            for ds, cfg in zip(self.predict_on, self.predict_config):
                self.predict_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="predict",
                    **cfg
                ))

    def train_dataloader(self):
        loader_list = []
        for ds in self.train_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.batch_size // len(self.train_on),
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=self.pin_memory,
            )
            loader_list.append(loader)
        return loader_list

    def val_dataloader(self):
        loader_list = []
        for ds in self.val_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
            loader_list.append(loader)
        return loader_list

    def test_dataloader(self):
        loader_list = []
        for ds in self.test_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
            loader_list.append(loader)
        return loader_list

    def predict_dataloader(self, shuffle=False):
        loader_list = []
        for ds in self.predict_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.test_batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
            loader_list.append(loader)
        return loader_list

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            tmp_batch = {k: list(map(itemgetter(k), batch)) for k in set(chain.from_iterable(batch))}
            return {k: torch.cat(v, dim=0) if k != 'filename' else [
                item for sublist in v for item in sublist] for k, v in tmp_batch.items()}
        else:
            return batch
