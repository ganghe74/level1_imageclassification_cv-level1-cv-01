from torchvision import datasets, transforms
from base import BaseDataLoader


class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train', transform=None):
        self.mode = mode
        self.files = files
        if mode == 'train':
            self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.mode == 'train':
            img = Image.open('train_imgs/' + self.files[i])

            if self.transform:
                img = self.transform(img)

            return {
                'img': torch.tensor(img, dtype=torch.float32).clone().detach(),
                'label': torch.tensor(self.labels[i], dtype=torch.long)
            }
        else:
            img = Image.open('test_imgs/' + self.files[i])
            if self.transform:
                img = self.transform(img)

            return {
                'img': torch.tensor(img, dtype=torch.float32).clone().detach(),
            }
            

class MaskDataLoader(BaseDataLoader):
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