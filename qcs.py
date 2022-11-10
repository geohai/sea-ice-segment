# import os

# import pandas as pd
# import torch
# from matplotlib import pyplot as plt
# import numpy as np
# from datasets import RasterDataset, GeoDataModule
# from torchvision.transforms import Normalize
# from models import DeepLabLikeMultiStream

# num_streams = 1
# num_classes = 23

# root_dir = os.path.normpath("E:/rafael/data/Extreme_Earth/")
# fname_in = os.path.join(root_dir, "EE_IO.csv")
# n_samples_per_input = 50
# val_prop = 0.2
# batch_size = 4
# crop_len = 20_000
# overlap = 0.75
# norms = None
# seed=0

# mean = [-13.50843779, -27.5089461, 34.06052272]
# std = [5.18822291, 4.53624699, 7.83304411]

# transforms = None
# norms = {}
# norms['input'] = Normalize(mean, std)

# df = pd.read_csv(fname_in)
# # maybe this is necessary
# # # add full path to df:
# # for col in df.columns:
# #     df[col] = root_dir+'/'+df[col]

# rdset = RasterDataset(df,
#                       n_samples_per_input=n_samples_per_input,
#                       crop_len=crop_len,
#                       overlap=overlap,
#                       transforms=transforms,
#                       norms=norms)

# for key, val in rdset.samples_dict.items():
#     fig, ax = plt.subplots(nrows=2)
#     if (val['input'].x.size > 0) & (val['input'].y.size > 0):
#         ax[0].imshow(val['input'].values[0,:,:])
#     if val['label'].x.size > 0:
#         ax[1].imshow(val['label'].values[0,:,:])
#     fig.show()
#     break

# full_dict = rdset.get_full_overlapping(0,[-34887-80_000, -34887+80_000,
#                                          -1116704-80_000, -1116704+80_000])

# # idxs = 0
# # for key, val in full_dict.items():
# #     fig, ax = plt.subplots(nrows=2)
# #     if (val['input'].x.size > 0) & (val['input'].y.size > 0):
# #         ax[0].imshow(val['input'].values[0,:,:])
# #     if val['label'].x.size > 0:
# #         ax[1].imshow(val['label'].values[0,:,:])
# #     fig.show()
# #     print(val['label'].x.mean(), val['label'].y.mean())
    
# #     idxs+=1
# #     if idxs==100:
# #         break


# rdm = GeoDataModule(df=df,
#                     n_samples_per_input=n_samples_per_input,
#                     crop_len=crop_len,
#                     overlap=overlap,
#                     transforms=transforms,
#                     norms=norms,
#                     seed=seed, 
#                     batch_size=batch_size, 
#                     val_prop=val_prop)
# rdm.setup(None)                      

# print(len(rdm.train_dataloader()))
# print(len(rdm.val_dataloader()))
# for sample in rdm.train_dataloader():
#     fig, ax = plt.subplots(nrows=2)
#     i=0
#     print(sample['input'].shape, sample['input'].max())
#     for key, val in sample.items():
#         ax[i].imshow(np.swapaxes(val[0,0,:,:].numpy(),0,-1))
#         i+=1
#     fig.show()

print('done')


import rioxarray
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
fnames_in = ['E:/rafael/data/Extreme_Earth/denoised_resampled/20180116t075430.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180213t175444.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180313t181225.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180417t074606.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180515t174633.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180612t180423.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180717t073809.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180814t075344.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20180911t175548.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20181016t072958.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20181113t074529.tif',
            'E:/rafael/data/Extreme_Earth/denoised_resampled/20181218t075437.tif']

for fname_in in fnames_in:
    rarr = rioxarray.open_rasterio(fname_in, masked=True)

    fig, ax = plt.subplots()
    ax.imshow(rarr[0].data, origin='lower')
    ax.set_title(fname_in)
    fig.show()

    b1 = rarr[0].data

    # assume all bands have the same mask:
    bmask = np.isnan(rarr[0].data).astype(int)
    # mark nan as zero
    bands = np.nan_to_num(rarr[0].data)

    # find the first pixel in row and col:
    y1 = (bmask[:,-1]==0).argmax()
    x1 = (bmask[0,:]==0).argmax() 

    angle = np.arctan2(y1, bmask.shape[1]-x1) * 180 / np.pi

    brot = ndimage.rotate(bands, angle, mode = 'nearest', reshape=True)
    #maskrot = ndimage.rotate(bmask, angle, mode = 'wrap', reshape=True)



    fig, ax = plt.subplots()
    ax.imshow(brot, origin='lower')
    ax.set_title(f'{fname_in} - {angle}')
    fig.show()
    print(fname_in)

print('done')