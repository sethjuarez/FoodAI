from modules import FoodModel, FoodDataModule
from pytorch_lightning import Trainer

if __name__ == '__main__':
    dm = FoodDataModule(data_dir='../data/food')
    dm.setup()
    model = FoodModel(classes=dm.classes)
    trainer = Trainer(gpus=1)
    trainer.fit(model, dm)