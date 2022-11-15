import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler


def plot_input():
    
    root_dir = os.path.normpath('E:/rafael/data/Extreme_Earth/denoised_resampled')
    for fname in Path(root_dir).rglob('*.tif*'):
        print(fname)
        da = rioxarray.open_rasterio(fname, masked=True)

        # clip 2-98% and rescale
        scaler = MinMaxScaler()
        for idx, band in enumerate(da.data):
            clip_min, clip_max = np.nanpercentile(band, [2,98])
            da.data[idx] = scaler.fit_transform(np.clip(band, clip_min, clip_max).reshape(-1,1)).reshape(band.shape)

        fig, ax = plt.subplots(figsize=(5,5))
        da.plot.imshow(ax=ax, rgb='band')
        ax.set_title('')
        ax.axis('off')
        fig.tight_layout()

        fig.savefig(os.path.join(root_dir, f'{Path(fname).stem}.png'), dpi=400)

        plt.close('all')

def plot_results_or_label():

    root_dirs = [os.path.normpath('E:/rafael/data/Extreme_Earth/results/v3/poly_type_wland/freeze/experiment3'), 
                 os.path.normpath('E:/rafael/data/Extreme_Earth/results/v3/poly_type_wland/all/experiment2'), 
                 os.path.normpath('E:/rafael/data/Extreme_Earth/results/v3/SA_wland/melt/experiment1'), 
                 os.path.normpath('E:/rafael/data/Extreme_Earth/results/v3/SA_wland/melt/experiment3'), 
                 os.path.normpath('E:/rafael/data/Extreme_Earth/results/v3/SD_wland-nopretrain/all - random initialization/experiment-notpretrained3'), 
                 #os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/SD_wland'),
                 #os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/SA_wland'),
                 #os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/poly_type_wland'),
                 ]

    for root_dir in root_dirs:
        for fname in Path(root_dir).rglob('*.tif*'):

            if Path(fname).stem.startswith('class-') or Path(fname).stem.startswith('correct-') or Path(fname).stem.startswith('seaice'):
                print(fname)
                da = rioxarray.open_rasterio(fname, masked=True)

                if 'poly_type' in os.path.normpath(fname):
                    cmap = ListedColormap([(51/255, 153/255, 255/255), (255/255, 125/255, 7/255), (196/255, 196/255, 196/255)])
                
                elif ('SA' in os.path.normpath(fname)) or ('SD' in os.path.normpath(fname)):
                    cmap = ListedColormap([(253/255, 204/255, 224/255), 
                                        (152/255, 111/255, 196/255), 
                                        (228/255, 0/255, 217/255), 
                                        (250/255, 243/255, 13/255), 
                                        (231/255, 61/255, 4/255), 
                                        (51/255, 153/255, 255/255), 
                                        (196/255, 196/255, 196/255)])           

                if Path(fname).stem.startswith('correct-'):
                    cmap = ListedColormap([(220/255, 16/255, 203/255), (0/255, 255/255, 0/255), (0/255, 255/255, 0/255)])

                fig, ax = plt.subplots(figsize=(5,5))
                da[0].plot.imshow(ax=ax, 
                                  vmin=0,
                                  vmax=len(cmap.colors)-1,
                                  add_colorbar=False, 
                                  interpolation='none', 
                                  cmap=cmap)
                ax.set_title('')
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(root_dir, f'{Path(fname).stem}.png'), dpi=400)

                plt.close('all')

if __name__ == '__main__':
    
    #plot_input()
    plot_results_or_label()

