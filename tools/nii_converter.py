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
        os.makedirs(target_dir)

    wrong_list = []
    with tqdm(total=len(nrrd_files)) as pbar:
        pbar.set_description('Processing')
        for index, nrrd_file in enumerate(nrrd_files, 1):
            nrrd_path = os.path.join(source_dir, nrrd_file)
            nrrd_data, nrrd_header = nrrd.read(nrrd_path)
            if nrrd_header['space directions'].shape != (3, 3) or nrrd_header['space origin'].shape != (3,):
                wrong_list.append(nrrd_file)
                pbar.update(1)
                continue
            affine_matrix = np.eye(4)
            affine_matrix[0:3, 0:3] = nrrd_header['space directions']
            affine_matrix[0:3, 3] = nrrd_header['space origin']
            nii_data = nib.Nifti1Image(nrrd_data, affine_matrix)
            nii_filename = os.path.splitext(nrrd_file)[0] + '.nii.gz'
            nib.save(nii_data,  os.path.join(target_dir, nii_filename))

            pbar.update(1)
    print('All nrrd files are converted to nii!')
    for i in wrong_list:
        print(i)


if __name__ == '__main__':
    main()
