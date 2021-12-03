import os
import sys
import logging

import torch
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Spacingd,
    ScaleIntensityd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    CropForegroundd
)

from models import VNet

data_dir = ""


# data directory:
# patient1.nii.gz
# patient1.seg.nii.gz
# ...
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = sorted(
        i for i in os.listdir(data_dir) if 'seg' not in i
    )
    segments = sorted(
        i for i in os.listdir(data_dir) if 'seg' in i
    )
    data_dicts = [
        {'img': img, 'seg': seg} for img, seg in zip(images, segments)
    ]
    keys = ['img', 'seg']
    train_transforms = Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Spacingd(keys=keys, pixdim=()),
            ScaleIntensityd(keys=['img']),
            CropForegroundd(keys=keys, source_key='img'),
            # prefer to pick a foreground voxel as a center
            RandCropByPosNegLabeld(keys=keys,
                                   label_key='seg',
                                   spatial_size=(150, 150, 150),
                                   pos=2,
                                   neg=1),
            EnsureTyped(keys=keys)
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Spacingd(keys=keys, pixdim=()),
            ScaleIntensityd(keys=['img']),
            CropForegroundd(keys=keys, source_key='img'),
            EnsureTyped(keys=keys)
        ]
    )

    # one fold evaluation
    for index in range(len(data_dicts)):
        val_files = data_dicts[index]
        train_files = data_dicts[: index] + data_dicts[index + 1:]

        train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=1.0, num_workers=3
        )
        train_loader = DataLoader(train_ds,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=3,
                                  )
        val_ds = CacheDataset(
            data=val_files, transform=val_transforms,
            cache_rate=1.0, num_workers=3
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=3)

        device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')

        model = VNet().to(device)
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        dice_metric = DiceMetric(include_background=False,
                                 reduction='mean')
        max_epochs = 600
        val_interval = 2

        for epoch in range(max_epochs):
            print("-" * 20)
            print(f"epoch {epoch + 1} / {max_epochs}")
            epoch_loss = 0
            step = 0
            model.train()
            for data in train_loader:
                step += 1
                inputs, labels = (
                    data['img'].to(device),
                    data['seg'].to(device),
                )
                optimizer.zero_grad()
                pred = model(inputs)
                loss = loss_function(pred, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f'{step}/{len(train_ds) // train_loader.batch_size},  '
                      f'train_loss: {loss.item():.4f}')
            epoch_loss /= step
            print(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')