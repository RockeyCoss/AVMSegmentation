# extract nrrd from mrb files
import argparse
import os
import shutil
import zipfile

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
    mrb_files = os.listdir(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with tqdm(total=len(mrb_files)) as pbar:
        pbar.set_description('Processing')
        for index, mrb_file in enumerate(mrb_files, 1):
            zip_path = os.path.join(source_dir, mrb_file)
            zip_folder = zipfile.ZipFile(zip_path)
            nrrd_list = [i for i in zip_folder.namelist() if i.endswith('.nrrd')]
            assert len(nrrd_list) == 2
            for file in nrrd_list:
                if 'seg' in file:
                    new_file_name = f'patient{index}.seg.nrrd'
                else:
                    new_file_name = f'patient{index}.nrrd'
                source = zip_folder.open(file)
                target = open(os.path.join(target_dir, new_file_name), 'wb')
                with source, target:
                    shutil.copyfileobj(source, target)
                # set window's size
                # if 'seg' not in new_file_name:
                #     img, _ = nrrd.read(os.path.join(target_dir, new_file_name))
                #     img = window_select(img)

            pbar.update(1)
    print('All mrb files are extracted!')


if __name__ == '__main__':
    main()