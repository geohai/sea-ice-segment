"""computes class weights for a collection of rasters
"""

import numpy as np
import rioxarray
from matplotlib import pyplot as plt
from tqdm import tqdm


def compute_class_weights(fnames, show_count=True):
    """Computes class weights (e.g., 
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html?highlight=weights) 
    of a collection of raster files

    Args:
        fnames (iterable): iterable containing the complete names of the raster files
        show_count (boolean): plots the histogram for rasters if True
    """
    counts_per_raster = []
    number_of_classes = 0
    
    print('==> Computing class weights...')
    for fname in tqdm(fnames):
        raster = rioxarray.open_rasterio(fname, masked=True)
        
        for idx, band in enumerate(raster):
            counts_per_raster.append(np.bincount(band.data[~np.isnan(band.data)].ravel().astype(int)))

            if len(counts_per_raster[-1]) > number_of_classes:
                number_of_classes = len(counts_per_raster[-1])
  
    class_count = [0 for _ in range(number_of_classes)]

    for bin_counter in counts_per_raster:
        for idx, bin_val in enumerate(bin_counter):
            class_count[idx]+=bin_val

    print(f'\nclass count: {class_count}')

    class_weights = np.sum(class_count)/(len(class_count) * np.array(class_count))
    print(f'\nclass weights: {class_weights}')
    
    if show_count:
        print('==> Generating images...')

        fig, ax = plt.subplots(ncols=2)
        ax[0].bar(x=range(len(class_count)), height=class_count)
        ax[0].set_title(f'Class Count')

        ax[1].bar(x=range(len(class_weights)), height=class_weights)
        ax[1].set_title(f'Class Weights')


        fig.show()

    return class_count, class_weights


if __name__ == '__main__':
    import os

    import pandas as pd
    root_dir = os.path.normpath("E:/rafael/data/Extreme_Earth/labels_rasterized/SA")
    fnames_selected = [ 'seaice_s1_20180213t175444-SW-SA.tif',
                        'seaice_s1_20180313t181225-SA.tif',
                        'seaice_s1_20180417t074606-SA.tif',
                        'seaice_s1_20180515t174633-SW-SA.tif',
                        'seaice_s1_20180612t180423-SA.tif',
                        'seaice_s1_20180814t075344-SW-SA.tif',
                        'seaice_s1_20180911t175548-SA.tif',
                        'seaice_s1_20181016t072958-SA.tif',
                        'seaice_s1_20181113t074529-SA.tif',
                        'seaice_s1_20181218t075437-SW-SA.tif']

    fnames = [os.path.join(root_dir, fname) for fname in fnames_selected]

    
    compute_class_weights(fnames,
                          )

    input("Press Enter to finish...")    
