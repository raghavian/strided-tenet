import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from kornia.color import RgbToGrayscale
import pdb
import numpy as np
import itertools
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fold2d(x,b=1):
    """
    Torch fold does not invert unfold! So a sloppy version that does the job.
    """
    d0 = int(np.sqrt(x.shape[0] / b))
    h = x.shape[-1]
    w = x.shape[-2]
    y = torch.zeros(b,int(d0*h),int(d0*w)).to(device)
    x = x.view(b,d0,d0,h,w)
    for i in range(d0):
        for j in range(d0):
            y[:,i*h:(i+1)*h,j*w:(j+1)*w] = x[:,i,j]
    return y


def squeeze(x,kernel=3):
    """
    Stack neighbourhood information per pixel
    along feature dimension
    """
    k = kernel // 2
    idx = [list(i) for i in itertools.product(range(-k,k+1),range(-k,k+1))]
    
    xPad = torch.zeros(x.shape[0]+kernel-1,x.shape[1]+kernel-1)
    xPad[k:-k,k:-k] = x
    
    xSqueezed = [torch.roll(xPad,shifts=(i[0],i[1]),dims=(0,1)) for i in idx]
    xSqueezed = torch.stack(xSqueezed)
    
    return xSqueezed[:,k:-k,k:-k]

class lungCXR(Dataset):
    def __init__(self, split='Train', data_dir = './',
            fold=0,transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        folds = [0,1,2,3]
        folds.remove(fold)
        if split == 'Valid':
            self.data, self.targets = torch.load(data_dir+'Fold'+repr(folds[0])+'.pt')
        elif split == 'Train':
            data0, targets0 = torch.load(data_dir+'Fold'+repr(folds[1])+'.pt')
            data1, targets1 = torch.load(data_dir+'Fold'+repr(folds[2])+'.pt')
            self.data = torch.cat((data0,data1),dim=0)
            self.targets = torch.cat((targets0,targets1),dim=0)
        else:
            self.data, self.targets = torch.load(data_dir+'Fold'+repr(fold)+'.pt')
        self.targets = self.targets.type(torch.FloatTensor)        
        self.data = self.data.squeeze().unsqueeze(3)
        self.targets = self.targets.squeeze().unsqueeze(3)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image.numpy(),mask=label.numpy())
            image = transformed["image"]
            label = transformed["mask"]
        return image, label.squeeze()

class MoNuSeg(Dataset):
    def __init__(self, split='Train', data_dir = './',transform=None):
        super().__init__()
        gray = RgbToGrayscale()
        self.data_dir = data_dir
        self.transform = transform
        self.data, self.targets = torch.load(data_dir+split+'.pt')
        self.data = self.data.permute(0,3,1,2)[:,[0]].squeeze().unsqueeze(3)

        self.targets = self.targets.squeeze().unsqueeze(3)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image.numpy(),mask=label.numpy())
            image = transformed["image"]
            label = transformed["mask"]

        return image, label.squeeze()


