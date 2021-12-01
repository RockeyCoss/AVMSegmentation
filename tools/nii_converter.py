# convert nrrd to nii format
import argparse
import os
import nrrd
import nibabel as nib
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('target_dir')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    nrrd_files = os.listdir(source_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    with tqdm(total=len(nrrd_files)) as pbar:
        pbar.set_description('Processing')
        for index, nrrd_file in enumerate(nrrd_files, 1):
            nrrd_path = os.path.join(source_dir, nrrd_file)
            nrrd_data, nrrd_header = nrrd.read(nrrd_path)

            nii_data = nib.Nifti1Image(nrrd_data, np.eye(4))
            nii_filename = os.path.splitext(nrrd_file)[0] + '.nii.gz'
            nib.save(nii_data,  os.path.join(target_dir, nii_filename))

            pbar.update(1)
    print('All nrrd files are converted to nii!')


if __name__ == '__main__':
    main()
