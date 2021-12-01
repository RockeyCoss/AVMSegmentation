import os

import torch
from monai.utils import first, set_determinism

data_dir = ""
# data directory:
# patient1.nii.gz
# patient1.seg.nii.gz
# ...
images = sorted(
    i for i in os.listdir(data_dir) if 'seg' not in i
)
labels = sorted(
    i for i in os.listdir(data_dir) if 'seg' in i
)

