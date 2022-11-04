import re
from torchvision import datasets
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import glob
from PIL import Image
import albumentations as A
from torchvision import transforms as T
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2
from torch.autograd import Variable
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import random

filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
masklabels = [1, 0, 0, 0, 0, 0, 2]

TOPCROP = A.Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512)
ONEOF = A.OneOf([
    A.Compose([]),
    A.CoarseDropout(always_apply=True, p=1.0, max_holes=60, max_height=10, max_width=10, min_holes=50, min_height=3, min_width=3, fill_value=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), mask_fill_value=None),
    A.CLAHE(always_apply=True, p=1.0, clip_limit=(10, 30), tile_grid_size=(60, 60)),
    A.Perspective(always_apply=True, p=1.0, scale=(0.13, 0.17), keep_size=1, pad_mode=0, pad_val=(0, 0, 0), mask_pad_val=0, fit_output=0, interpolation=0),
    A.GridDistortion(always_apply=True, p=1.0, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=random.randint(0,4), border_mode=random.randint(0,4), value=(0, 0, 0), mask_value=None, normalized=False),
    A.CenterCrop(300,300,p=1)
], p=1.)
TT = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224))
])

def tf(im):
    im = np.array(im)
    im = TOPCROP(image=im)['image']
    im = ONEOF(image=im)['image']
    im = TT(im)
    return im

class MaskTrainDataset(Dataset):
    def __init__(self, root, transform, training=True):
        self.root = root
        self.is_train = training
        self.transform = transform
        
        ### augmentations 달라지면 수정해줘야함
        self.aug_name = '_OD'
        self.aug_filenames = ['incorrect_mask'+self.aug_name, 'mask1'+self.aug_name, 'mask2'+self.aug_name, 'mask3'+self.aug_name, 'mask4'+self.aug_name, 'mask5'+self.aug_name, 'normal'+self.aug_name]
        self.aug_masklabels = [1, 0, 0, 0, 0, 0, 2]
        ###
        
        self.paths = []
        if self.is_train:
            self.df = pd.read_csv(os.path.join(root, 'train', 'train.csv'))
            self.labels = []
            for path in self.df['path']: # path == 000002_female_Asian_52
                _, gender, _, age = path.split('_')
                age = int(age)
                gender = 0 if gender == 'male' else 1
                if age < 30:
                    age = 0
                elif age < 55:
                    age = 1
                else:
                    age = 2

                filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
                for file, mask in zip(filenames, masklabels): # file == "incorrect_mask", mask == 1
                    p = os.path.join(root, 'train', 'images', path, file+'*')
                    self.paths.extend(glob.glob(p))
                    self.labels.append(mask * 6 + gender * 3 + age)
                    
                    # for file, aug_mask in zip(aug_filenames, aug_masklabels): # file == "incorrect_mask_CC"
                    
                    if age == 2:
                        aug_p = os.path.join(root, 'off_aug', 'images', path, file+'_R'+'*')
                        self.paths.extend(glob.glob(aug_p))
                        self.labels.append(mask * 6 + gender * 3 + age)
                        
                        aug_p = os.path.join(root, 'off_aug', 'images', path, file+'_HF'+'*') 
                        self.paths.extend(glob.glob(aug_p))
                        self.labels.append(mask * 6 + gender * 3 + age)
                    
        else:
            self.df = pd.read_csv(os.path.join(root, 'eval', 'info.csv'))
            self.paths = [os.path.join(root, 'eval', 'images', img_id) for img_id in self.df.ImageID]
                    

    def __getitem__(self, index):
        image = np.array(Image.open(self.paths[index]))
        # image = cv2.imread(self.paths[index])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.is_train:
            label = self.labels[index]
            return Variable(image.float()), label
            
        else:
            return Variable(image.float())
        
    def __len__(self):
        return len(self.paths)
    
class MaskGlobDataset(Dataset):
    def __init__(self, root, transform, train=True, paths=None):
        """
        csv 없이 파일 경로에서 라벨을 추출하는 데이터셋
        OfflineAug를 위해 제작함
        Args:
            root: 이미지가 들어있는 최상위 디렉터리
                  ex) '/opt/ml/input/data/train'
            transform:
            train:
        """
        self.root = root = Path(root)
        self.train = train
        self.transform = transform

        self.paths = []
        self.labels = []

        if paths is None:
            files = root.glob('**/*')
            self.paths = [f for f in files if self._is_image(f)]
        else:
            for path in paths:
                # path == '000001_female_Asian_45'
                files = (root / 'images' / path).glob('*.*')
                files = [f for f in files if self._is_image(f)]
                self.paths.extend(files)
                
                # 60aug_image path 추가
                if self._have_aug_image(path):
                    files = (root / '60aug-images' / path).glob('*.*')
                    files = [f for f in files if self._is_image(f)]
                    self.paths.extend(files)
                
                # # 60세 미만 and mask : not wear or incorrect Aug path 추가
                # if self._get_age(path) < 60:
                #     files = (root / 'notwear-incorrect-aug-images' / path).glob('*.*')
                #     files = [f for f in files if self._is_image(f)]
                #     self.paths.extend(files)

        if train:
            for p in self.paths:
                self.labels.append(self._parse(p))

    def _have_aug_image(self, path):
        path = str(path)
        age = int(path.split('_')[3])
        if age == 60:
            return True
        else:
            False

    def _is_image(self, path):
        exts = ['jpg', 'jpeg', 'png']
        p = str(path)
        return '._' not in p and any(p.endswith(ext) for ext in exts)


    def _parse(self, p):
        """
        path를 파싱해 라벨 리턴
        """
        p = str(p)
        match = re.search('(.+)_Asian_(\d+)/(.*)[\.-]', p)
        if match and len(match.groups()) == 3:
            gender, age, mask = match.groups()
            gender = 0 if gender == 'male' else 1
            age = int(age)
            if age < 30:
                age = 0
            elif age < 60:
                age = 1
            else:
                age = 2

            if mask.startswith('normal'):
                mask = 2
            elif mask.startswith('incorrect'):
                mask = 1
            else:
                mask = 0
            return mask * 6 + gender * 3 + age
        else:
            raise Exception(f'Cannot parsing label from the path: {p}')
    
    def _get_age(self, path):
        '''
        # path == '000001_female_Asian_45'
        '''
        path = str(path)
        age = int(path.split('_')[3])
        return age

    def __getitem__(self, index):
        image = Image.open(self.paths[index])
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = self.labels[index]
            return image, label
        return image, -1

    def __len__(self):
        return len(self.paths)
    
class MaskDataLoader(BaseDataLoader):
    """
    Competition Mask Dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((256, 192)),
        #     albumentations.RandomRain(slant_lower=-20,slant_upper=20,drop_length=20,drop_width=1,drop_color=(200,200,200),blur_value=1,brightness_coefficient=0.9,rain_type=None,always_apply=True,p=0.5)
        # ])
        
        trsfm = A.Compose([
            A.Resize(256, 192), 
            # albumentations.RandomRain(slant_lower=-20,slant_upper=20,drop_length=20,drop_width=1,drop_color=(200,200,200),blur_value=1,brightness_coefficient=0.9,rain_type=None,always_apply=True,p=0.5),
            ToTensorV2()
        ])
        
        
        # self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        self.dataset = MaskTrainDataset(data_dir, trsfm, training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MaskSplitLoader(DataLoader):
    """
    Competition Mask DataLoader. Split by profile
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, dataset='default'):
        # default transform
        # train_trsfm = T.Compose([
        #     T.ToTensor(),
        #     T.Resize((256, 192))
        # ])
        train_trsfm = tf
        valid_trsfm = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224))
        ])
        
        self.data_dir = data_dir = Path(data_dir) # 이게 뭐임?
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

        self.trainset = MaskGlobDataset(data_dir, train_trsfm, paths=self.paths[train_idx])
        self.validset = MaskGlobDataset(data_dir, valid_trsfm, paths=self.paths[valid_idx])
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

def age_parse(age):
    if age < 30:
        return 0
    elif age < 60:
        return 1
    else:
        return 2

