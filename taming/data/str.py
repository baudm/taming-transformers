import string

from torch.utils.data import Dataset
from torchvision import transforms as T

from taming.data.dataset import hierarchical_dataset


class opt:
    imgH = 224
    imgW = 224
    data_filtering_off = True
    PAD = False
    sensitive = True
    character = string.printable[:-6]
    rgb = False
    batch_max_length = 25


class STRTrain(Dataset):

    def __init__(self, config=None) -> None:
        super().__init__()
        transforms = T.Compose([
            T.Resize((opt.imgH, opt.imgW), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            # model expects data in channel-last format
            T.Lambda(lambda x: x.permute(1, 2, 0))
        ])
        root = '../data_lmdb_release/training'
        self.dataset = hierarchical_dataset(root, opt, transform=transforms)[0]

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {'image': img}

    def __len__(self):
        return len(self.dataset)


class STRValidation(Dataset):

    def __init__(self, config=None) -> None:
        super().__init__()
        transforms = T.Compose([
            T.Resize((opt.imgH, opt.imgW), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
            # model expects data in channel-last format
            T.Lambda(lambda x: x.permute(1, 2, 0))
        ])
        root = '../data_lmdb_release/validation'
        self.dataset = hierarchical_dataset(root, opt, transform=transforms)[0]

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {'image': img}

    def __len__(self):
        return len(self.dataset)
