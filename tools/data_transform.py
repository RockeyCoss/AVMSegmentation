# window size transform
import argparse
import os
import shutil

import nrrd
import numpy as np
from tqdm import tqdm


def window_select(img, center=77, width=800):
    img_min = center - width//2
    img_max = center + width//2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    intensity_step = 255.0 / (img_max - img_min)
    img_new = np.round((img - img_min) * intensity_step).astype(np.int8)
    return img_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('target_dir')
    args = parser.parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    with tqdm(total=len(os.listdir(source_dir))) as pbar:
        for i in os.listdir(source_dir):
            if 'seg' not in i:
                img, header = nrrd.read(os.path.join(source_dir, i))
                img = window_select(img, width=177)
                nrrd.write(os.path.join(target_dir, i), img,header)
            else:
                shutil.copyfile(os.path.join(source_dir, i), target_dir)
            pbar.update(1)
    print("all done")