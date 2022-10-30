from torch.utils.data import DataLoader
from torchvision import transforms
from base import BaseDataLoader
from .dataset import MaskDataset, MaskGlobDataset
from glob import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class MaskDataLoader(BaseDataLoader):
    """
    Competition Mask DataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, dataset='default'):
        # default transform
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((320, 256))
        ])

        self.data_dir = data_dir
        self.dataset = self._get_dataset(dataset, data_dir, trsfm, training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    def _get_dataset(self, dataset, data_dir, trsfm, train):
        if dataset == 'glob':
            return MaskGlobDataset(data_dir, trsfm, train)
        return MaskDataset(data_dir, trsfm, train)



class MaskSplitLoader(DataLoader):
    """
    Competition Mask DataLoader. Split by profile
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, dataset='default'):
        # default transform
        train_trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 192))
        ])
        valid_trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 192))
        ])

        self.data_dir = data_dir = Path(data_dir)
        self.df = pd.read_csv(data_dir / 'train.csv')
        self.paths = self.df['path'].to_numpy()
        self.labels = (self.df['gender'].map({'male': 0, 'female': 1}) * 3 + self.df['age'] // 3).to_numpy()

        s = StratifiedShuffleSplit(n_splits=1, test_size=validation_split)
        s.get_n_splits()
        train_idx, valid_idx = next(s.split(self.paths, self.labels))

        self.trainset = MaskGlobDataset(data_dir, train_trsfm, paths=self.paths[train_idx])
        self.validset = MaskGlobDataset(data_dir, train_trsfm, paths=self.paths[valid_idx])
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