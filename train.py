import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model import Deeplabv3Plus
from dataset import PersonSegmentDataModule


from config import (BATCH_SIZE, DEVICE, LEARNING_RATE, NUM_EPOCHS,
                    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR)


model = Deeplabv3Plus(num_classes=1, learning_rate=1e-5).to(DEVICE)


val_checkpoint = ModelCheckpoint(
    dirpath="checkpoints/",
    monitor= "val_loss",
    mode="min",
    filename='{epoch}-{step}-{val_loss:.1f}',
    save_top_k=1
)

model_checkpoint = ModelCheckpoint(
    dirpath="checkpoints/",
    monitor= "epoch",
    mode="max",
    filename="{epoch}",
    save_top_k=1
)


class LearningRateFinderMilestones(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


lr_monitor = LearningRateMonitor(logging_interval='epoch')

logger = TensorBoardLogger(name="Deep Lab",save_dir = "logs")

trainer = pl.Trainer(accelerator="gpu",
                     strategy="ddp_find_unused_parameters_true",
                     logger=logger,
                     devices=2,
                     min_epochs=3,
                     max_epochs=NUM_EPOCHS,
                     precision=16,
                     val_check_interval= 0.5, # how many times we want to validate during an epoch
                     callbacks=[val_checkpoint,model_checkpoint, LearningRateFinderMilestones(milestones=(2,8)),lr_monitor])


data_module = PersonSegmentDataModule(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    BATCH_SIZE
)

trainer.fit(model, data_module)
trainer.validate(model, data_module)