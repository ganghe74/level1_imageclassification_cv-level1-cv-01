from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
from PIL import Image

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
masklabels = [1, 0, 0, 0, 0, 0, 2]

class MaskTrainDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        self.df = pd.read_csv(os.path.join(root, 'train', 'train.csv'))

        self.paths = []
        self.labels = []
        for path in self.df['path']:
            _, gender, _, age = path.split('_')
            age = int(age)
            gender = 1 if gender == 'male' else 0
            if age < 30:
                age = 0
            elif age < 60:
                age = 1
            else:
                age = 2

            for file, mask in zip(filenames, masklabels):
                p = os.path.join(root, 'train', 'images', path, file+'*')
                self.paths.extend(glob.glob(p))
                self.labels.append(mask * 6 + gender * 3 + age)
                # self.labels.append((mask, gender, age))
                

    def __getitem__(self, index):
        image = Image.open(self.paths[index])

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.paths)


class MaskDataLoader(BaseDataLoader):
    """
    Competition Mask Dataset
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        # self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        self.dataset = MaskTrainDataset('/opt/ml/input/data', trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)