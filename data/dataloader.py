from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
    

    
############################################################
    
from torch.utils.data import Dataset
import data.util as Util
import os
import numpy as np
import scipy.io as sio

@register_dataset(name='acdc')
class ACDCDataset(Dataset):
    def __init__(self, root,  split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = root

        datapath = os.path.join(root, split)
        dataFiles = sorted(os.listdir(datapath))
        for isub, dataName in enumerate(dataFiles):
            self.imageNum.append(os.path.join(datapath, dataName))

        self.data_len = len(self.imageNum)
        self.fineSize = [128, 128, 32]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        dataPath = self.imageNum[index]
        data_ = sio.loadmat(dataPath)
        dataA = data_['image_ED']
        dataB = data_['image_ES']
        label_dataA = data_['label_ED']
        label_dataB = data_['label_ES']

        # if self.split == 'test':
        #     dataName = dataPath.split('/')[-1]
        #     data_ = sio.loadmat(os.path.join(self.dataroot, 'data_ED2ES', dataName))
        #     dataW = data_['image']
        #     nsample = dataW.shape[-1]
        # else:
        nsample = 7 # the average number of samples used in inferrence only 

        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()

        nh, nw, nd = dataA.shape
        desired_h, desired_w, desired_d = self.fineSize  # fineSize now includes depth


        pad_h = max(0, desired_h - nh)
        pad_w = max(0, desired_w - nw)
        pad_d = max(0, desired_d - nd)


        # Apply padding evenly on both sides, with any extra padding added to the bottom/right
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front        
        
        # Pad the images and labels if necessary
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            dataA = np.pad(dataA, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
            label_dataA = np.pad(label_dataA, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
            dataB = np.pad(dataB, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
            label_dataB = np.pad(label_dataB, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
        
        # Recalculate new heights and widths after padding
        nh, nw, nd = dataA.shape
        
        # Calculate start points for cropping (automatically handles indivisibility by using integer division)
        sh = max(0, (nh - desired_h) // 2)
        sw = max(0, (nw - desired_w) // 2)
        sd = max(0, (nd - desired_d) // 2)

        # Crop to the desired size for height, width, and depth
        dataA = dataA[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
        label_dataA = label_dataA[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
        dataB = dataB[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
        label_dataB = label_dataB[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]        
        

        [data, label] = Util.transform_augment([dataA, dataB], split=self.split, min_max=(-1, 1))

        # return {'S': data, 'T': label, 'SL': label_dataA, 'TL': label_dataB, 'nS':nsample, 'P':dataPath, 'Index': index}
        return data
    