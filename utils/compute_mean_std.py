"""computes mean and std of rasters 
"""

import numpy as np
import rioxarray
from matplotlib import pyplot as plt
from tqdm import tqdm


def compute_mean_std(fnames, n_channels, show_histogram=True, bins=100):
    """Computes mean and standard deviation of a collection of raster files

    Args:
        fnames (iterable): iterable containing the complete names of the raster files
        n_channels (int): number of bands
        show_histogram (boolean): plots the histogram for all bands if True
        bins (int): number of bins for histogram
    """
    dset_mean = np.zeros(n_channels)
    dset_std = np.zeros(n_channels)
    bands = [[] for _ in range(n_channels)]

    print('==> Computing mean and std...')
    for fname in tqdm(fnames):
        raster = rioxarray.open_rasterio(fname, masked=True)
        
        for idx, band in enumerate(raster):
            dset_mean[idx] += band.mean().item()
            dset_std[idx] += band.std().item()
            
            bands[idx].append(band)

    dset_mean = dset_mean/(len(fnames))
    dset_std = dset_std/(len(fnames))

    print(f'\nmean: {dset_mean}')
    print(f'std: {dset_std}')

    if show_histogram:
        print('==> Computing histograms...')
        for idx, band_list in enumerate(tqdm(bands)):
            x = np.concatenate([darray.to_numpy().ravel() for darray in band_list])
            x = x[~np.isnan(x)]

            fig, ax = plt.subplots()
            ax.hist(x, bins=bins)
            ax.set_title(f'Histogram for band {idx+1}')

            fig.show()

    return dset_mean, dset_std


if __name__ == '__main__':
    import os

    import pandas as pd
    root_dir = os.path.normpath("E:/rafael/data/Extreme_Earth/results/v1/poly_type_wland")
    fname_in = os.path.join(root_dir, "EE_IO-poly_type-melt.csv")
    df = pd.read_csv(fname_in)

    df = df.loc[df['dset'] == 'train'].copy()

    fnames = df['input'].to_list()
    n_channels = 3
    
    compute_mean_std(fnames,
                     n_channels,
                     )

    input("Press Enter to finish...")    
