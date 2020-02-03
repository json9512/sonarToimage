import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        
        self.image_path = f'{root}/camera_resized/'
        self.images = sorted([int(x.split(".")[0]) for x in os.listdir(self.image_path)])

        self.scan_path = f'{root}/resized_images/'
        self.scans = sorted([int(x.split(".")[0]) for x in os.listdir(self.scan_path)])

    def __getitem__(self, index):
        
        
        img = Image.open(f'{self.image_path}{self.images[index % len(self.images)]}.png')
        scan = Image.open(f'{self.scan_path}{self.scans[index % len(self.scans)]}.png')
        
        if np.random.random() < 0.5:
            img = Image.fromarray(np.array(img)[:, ::-1, :], "RGB")
            scan = Image.fromarray(np.array(scan)[:, ::-1, :], "RGB")
        
        img = self.transform(img)
        scan = self.transform(scan)
        
        return {"camera": img, "scan": scan}
        

    def __len__(self):
        return len(self.images)
    
class TestDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="val"):
        self.transform = transforms.Compose(transforms_)

        self.scan_path = f'{root}/'
        self.scans = sorted([int(x.split(".")[0]) for x in os.listdir(self.scan_path)])

    def __getitem__(self, index):
        
        scan = Image.open(f'{self.scan_path}{self.scans[index % len(self.scans)]}.png')
        scan = self.transform(scan)
        
        return {"scan": scan}
        
    def __len__(self):
        return len(self.scans)
    
