import os
import os.path as osp
import sys
import logging
from argparse import ArgumentParser

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Spacingd,
    ScaleIntensityd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    CropForegroundd, EnsureType, AsDiscrete, Orientationd
)

from core.models.builder import build_model
from core.utils import load_config, Logger

work_dir = ""


# data directory:
# patient1.nii.gz
# patient1.seg.nii.gz
# ...
def main():
    parser = ArgumentParser()
    parser.add_argument('config_dir')
    args = parser.parse_args()
    cfg = load_config(args.config_dir)
    data_dir = cfg['data_dir']
    batch_size = cfg['batch_size']
    patch_size = cfg['patch_size']
    if 'work_dir' not in cfg:
        work_dir = f'./work_dir/{osp.splitext(osp.basename(args.config_dir))[0]}'
    else:
        work_dir = cfg['work_dir']
    if not osp.exists(work_dir):
        os.makedirs(work_dir)
    logger = Logger(osp.join(work_dir, 'log.txt'))

    images = sorted(
        osp.join(data_dir, i) for i in os.listdir(data_dir) if 'seg' not in i
    )
    segments = sorted(
        osp.join(data_dir, i) for i in os.listdir(data_dir) if 'seg' in i
    )
    data_dicts = [
        {'img': img, 'seg': seg} for img, seg in zip(images, segments)
    ]
    keys = ['img', 'seg']
    train_transforms = Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Spacingd(
                keys=keys,
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=keys, axcodes="RAS"),
            ScaleIntensityd(keys=['img']),
            CropForegroundd(keys=keys, source_key='img'),
            # prefer to pick a foreground voxel as a center
            RandCropByPosNegLabeld(keys=keys,
                                   label_key='seg',
                                   spatial_size=patch_size,
                                   pos=2,
                                   neg=1,
                                   num_samples=4),
            EnsureTyped(keys=keys)
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Spacingd(
                keys=keys,
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=keys, axcodes="RAS"),
            ScaleIntensityd(keys=['img']),
            CropForegroundd(keys=keys, source_key='img'),
            EnsureTyped(keys=keys)
        ]
    )

    # one fold evaluation
    for index in range(len(data_dicts)):
        val_files = [data_dicts[index]]
        train_files = data_dicts[: index] + data_dicts[index + 1:]
        logger.print_and_log('-' * 10)
        logger.print_and_log('Fold Start')
        logger.print_and_log('-' * 10)
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=1.0, num_workers=4
        )
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True
                                  )
        val_ds = CacheDataset(
            data=val_files, transform=val_transforms,
            cache_rate=1.0, num_workers=4
        )
        val_loader = DataLoader(val_ds,
                                batch_size=batch_size, num_workers=4,
                                shuffle=False, pin_memory=True)

        device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')

        model = build_model(dict(type='UNet')).to(device)
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        dice_metric = DiceMetric(include_background=False,
                                 reduction='mean')
        max_epochs = 600
        val_interval = 2
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

        for epoch in range(max_epochs):
            logger.print_and_log("-" * 20)
            logger.print_and_log(f"epoch {epoch + 1} / {max_epochs}")
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
                logger.print_and_log(f'{step}/{len(train_ds) // train_loader.batch_size},  '
                                     f'train_loss: {loss.item():.4f}')
            epoch_loss /= step
            logger.print_and_log(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')

            # validation
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_seg = (
                            val_data['img'].to(device),
                            val_data['seg'].to(device),
                        )
                        roi_size = patch_size
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, model)
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                        dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()
                    logger.print_and_log(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    )
                    # torch.save(model.state_dict(), os.path.join(
                    #     work_dir, f"epoch{epoch + 1}:dice{metric:.4f}.pth"
                    # ))


if __name__ == '__main__':
    main()
