from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
from PIL import Image
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2
from torch.autograd import Variable

filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
masklabels = [1, 0, 0, 0, 0, 0, 2]


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
                elif age < 60:
                    age = 1
                else:
                    age = 2

                filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
                for file, mask in zip(filenames, masklabels): # file == "incorrect_mask", mask == 1
                    p = os.path.join(root, 'train', 'images', path, file+'*')
                    self.paths.extend(glob.glob(p))
                    self.labels.append(mask * 6 + gender * 3 + age)
                    
                    # for file, aug_mask in zip(aug_filenames, aug_masklabels): # file == "incorrect_mask_CC"
                    
                    # if age == 2:
                    #     aug_p = os.path.join(root, 'off_aug', 'images', path, file+'_R'+'*')
                    #     self.paths.extend(glob.glob(aug_p))
                    #     self.labels.append(mask * 6 + gender * 3 + age)
                        
                    #     aug_p = os.path.join(root, 'off_aug', 'images', path, file+'_HF'+'*') 
                    #     self.paths.extend(glob.glob(aug_p))
                    #     self.labels.append(mask * 6 + gender * 3 + age)
                    
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
        
        trsfm = albumentations.Compose([
            albumentations.Resize(256, 192), 
            albumentations.RandomRain(slant_lower=-20,slant_upper=20,drop_length=20,drop_width=1,drop_color=(200,200,200),blur_value=1,brightness_coefficient=0.9,rain_type=None,always_apply=True,p=0.5),
            ToTensorV2()
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