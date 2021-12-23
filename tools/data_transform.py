# window size transform
import argparse
import os
import shutil

import nrrd
import numpy as np
from tqdm import tqdm


def window_select(img, center=400, width=1000):
    img_min = center - width / 2
    img_max = center + width / 2
    img = img.astype(float)
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    intensity_step = 255.0 / (img_max - img_min)
    img_new = np.round((img - img_min) * intensity_step).astype(np.uint8)
    return img_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('target_dir')
    args = parser.parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with tqdm(total=len(os.listdir(source_dir))) as pbar:
        for i in os.listdir(source_dir):
            if 'seg' not in i:
                img, header = nrrd.read(os.path.join(source_dir, i))
                img = window_select(img)
                nrrd.write(os.path.join(target_dir, i), img, header)
            else:
                shutil.copy(os.path.join(source_dir, i), target_dir)
            pbar.update(1)
    print("all done")