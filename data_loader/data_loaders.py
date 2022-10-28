from torchvision import transforms
from base import BaseDataLoader
from .dataset import MaskDataset, MaskGlobDataset


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
        return MaskDataset(data_dir, trsfm, training)


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