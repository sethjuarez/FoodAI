import torch
import torch.nn as nn
from typing import List
import torch.optim as optim
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict

class FoodModel(pl.LightningModule):
    def __init__(self, classes: List[str], learning_rate: float = 0.1):
        super().__init__()
        self.lr = learning_rate
        self.classes = classes

        # model layers
        self.xfer = models.resnet18(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, len(self.classes))

    def forward(self, x):
        x = F.relu(self.xfer(x))
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

    @staticmethod
    def __accuracy(outputs, Y):
        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            return torch.sum(Y == preds).item() / (len(Y) * 1.0)

    def __step(self, batch):
        X, Y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, Y)
        return loss, self.__accuracy(outputs, Y)

    def training_step(self, batch, batch_idx):
        loss, acc = self.__step(batch)
        #self.log('loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.__step(batch)
        #self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return OrderedDict({
            'val_loss': loss,
            'val_acc': acc
        })

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), 
            lr=self.lr
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=7, 
            gamma=0.1
        )

        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items