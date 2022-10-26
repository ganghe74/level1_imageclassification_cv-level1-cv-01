from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
from PIL import Image

filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
masklabels = [1, 0, 0, 0, 0, 0, 2]
aug_masklabels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]

class MaskTrainDataset(Dataset):
    def __init__(self, root, transform, training=True):
        self.root = root
        self.is_train = training
        self.transform = transform
        
        self.paths = []
        if self.is_train:
            self.df = pd.read_csv(os.path.join(root, 'train', 'train.csv'))
            self.labels = []
            for path in self.df['path']:
                _, gender, _, age = path.split('_')
                age = int(age)
                gender = 0 if gender == 'male' else 1
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
                    for aug_mask in aug_masklabels:
                        aug_p = os.path.join(root, 'off_aug', 'images', path, file+'*')
                        self.paths.extend(glob.glob(aug_p))
                        self.labels.append(aug_mask * 6 + gender * 3 + age)
        else:
            self.df = pd.read_csv(os.path.join(root, 'eval', 'info.csv'))
            self.paths = [os.path.join(root, 'eval', 'images', img_id) for img_id in self.df.ImageID]
                    

    def __getitem__(self, index):
        image = Image.open(self.paths[index])

        if self.transform:
            image = self.transform(image)
        if self.is_train:
            label = self.labels[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.paths)


class MaskDataLoader(BaseDataLoader):
    """
    Competition Mask Dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((192, 256))
        ])
        # self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        self.dataset = MaskTrainDataset(data_dir, trsfm, training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


'''
mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(244),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
])

myvaltransform =transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
'''