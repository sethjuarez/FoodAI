import math
import torch
import pytorch_lightning as pl
from typing import List, Optional
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

class FoodDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", 
                 batch_size: int = 8, 
                 split: List[float] = [.7, .2, .1]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split = torch.tensor(split)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        # check split
        if len(self.split) != 3: raise Exception("split size should be 3 (train, val, test)")
        # normalize split
        self.split = self.split / self.split.sum()

        # using ImageFolder
        food_images = datasets.ImageFolder(self.data_dir, self.transform)
        self.classes = food_images.classes

        # set dims
        self.dims = tuple(food_images[0][0].shape)

        # counts/splits
        sz = len(food_images)
        train_sz = math.floor(self.split[0] * sz)
        val_sz = math.floor(self.split[1] * sz)
        test_sz = sz - train_sz - val_sz

        # split
        self.train, self.val, self.test = random_split(food_images, [train_sz, val_sz, test_sz])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
