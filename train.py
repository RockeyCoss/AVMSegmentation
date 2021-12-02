import os
import sys
import logging

import torch
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    ScaleIntensity,
    AddChannel,
    RandSpatialCrop,
    LoadImaged,
    AddChanneld,
    Spacingd,
    ScaleIntensityd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    CropForegroundd
)

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

        # model = 