import os
import numpy as np
from PIL import Image
import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class PersonSegmentData(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        super(PersonSegmentData, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", ".png"))

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"),
                        dtype=np.float32)  # l -> grayscale
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augemantations = self.transform(image=image, mask=mask)
            image = augemantations['image']
            mask = augemantations['mask']
        return image, mask


class PersonSegmentDataModule(pl.LightningDataModule):

    def __init__(self, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=32):
        super().__init__()
        self.train_image_dir = train_image_dir
        self.train_mask_dir = train_mask_dir
        self.val_image_dir = val_image_dir
        self.val_mask_dir = val_mask_dir
        self.batch_size = batch_size
        
        # Define transformations
        self.train_transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        self.val_transforms = A.Compose(
            [
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = PersonSegmentData(
            self.train_image_dir, self.train_mask_dir, transform=self.train_transform)
        self.val_dataset = PersonSegmentData(
            self.val_image_dir, self.val_mask_dir, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=20)
