from torch.utils.data import DataLoader
from torchvision import transforms as T
from base import BaseDataLoader
from .dataset import MaskDataset, MaskGlobDataset
from glob import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import albumentations as A
from random import choice


class MaskDataLoader(BaseDataLoader):
    """
    Competition Mask DataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, dataset='default'):
        # default transform
        trsfm = T.Compose([
            T.ToTensor(),
            # transforms.Normalize((0.5489362, 0.50472213, 0.48014935), (0.23510724, 0.24488226, 0.24451137)),
            T.Resize((256, 192))
        ])

        self.data_dir = data_dir
        self.dataset = self._get_dataset(dataset, data_dir, trsfm, training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    def _get_dataset(self, dataset, data_dir, trsfm, train):
        if dataset == 'glob':
            return MaskGlobDataset(data_dir, trsfm, train)
        return MaskDataset(data_dir, trsfm, train)

# [Origin, RandomCrop(300), CenterCrop(300),  CLAHE,  GridDistortion (찌그러트림), Perspective, GridDistortion+Crop] + CoarseDropout(CutOut)

TOPCROP = A.Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512)
ONEOF = A.OneOf([
    A.Compose([]),
    A.RandomCrop(300, 300, p=1),
    A.CenterCrop(300, 300, p=1),
    A.CLAHE(p=1),
    A.GridDistortion(p=1),
    A.Perspective(p=1),
    A.Compose([
        A.GridDistortion(p=1),
        A.RandomCrop(300, 300, p=1)
    ])
], p=1.)
CUTOUT = A.CoarseDropout()
TT = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224))
])

def tf(im):
    im = np.array(im)
    im = TOPCROP(image=im)['image']
    im = ONEOF(image=im)['image']
    im = CUTOUT(image=im)['image']
    im = TT(im)
    return im


class MaskSplitLoader(DataLoader):
    """
    Competition Mask DataLoader. Split by profile
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, dataset='default'):
        train_trsfm = tf
        valid_trsfm = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224))
        ])

        self.data_dir = data_dir = Path(data_dir)
        self.df = pd.read_csv(data_dir / 'train.csv')
        self.paths = self.df['path'].to_numpy()
        self.labels = (self.df['gender'].map({'male': 0, 'female': 1}) * 3 + self.df['age'] // 30).to_numpy()

        if validation_split == 0:
            train_idx = range(len(self.paths))
            valid_idx = []
        else:
            s = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=0)
            s.get_n_splits()
            train_idx, valid_idx = next(s.split(self.paths, self.labels))

        self.trainset = MaskGlobDataset(data_dir, train_trsfm, valid=False, paths=self.paths[train_idx])
        self.validset = MaskGlobDataset(data_dir, valid_trsfm, valid=True, paths=self.paths[valid_idx])
        self.n_samples = len(self.trainset)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        super().__init__(self.trainset, **self.init_kwargs)
    
    def _split_sampler(self):
        raise Exception('do not use _split_sampler')
    
    def split_validation(self):
        dl = DataLoader(self.validset, **self.init_kwargs)
        dl.n_samples = len(self.validset)
        return dl