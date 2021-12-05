model = dict(type='UNet')
patch_size = (10, 10, 10)
batch_size = 1
num_workers = 1
cache_rate=1.0
val_interval=50
max_epoch=500
data_dir = r'G:\data\nii'