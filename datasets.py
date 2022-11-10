"""
Dataset and DataLoader (pl.LightningDataModule) using raster files.
The classes here do not check for projection, dimensions, overlapping between inputs. 

"""
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rioxarray
import torch
from numpy.random import MT19937, RandomState, SeedSequence
from torch.utils.data import DataLoader, Dataset


class RasterDataset(Dataset):
    """Geological mapping dataset
    """

    def __init__(self,
                 df,
                 n_samples_per_input,
                 crop_len,
                 ignore_index,
                 transforms=None,
                 norms=None,
                 seed=123456789):
        """Class that reads raster using rasters into xarray objects using rioxarray. 
        The data is clipped according to the provided parameters and transformed into
        pytorch tensors. 

        Args:
            df (pd.DataFrame or string): input file names as path to file (csv) or pd.DataFrame.
                The dataframe is expected to have at least one column called "input". 
                Labels can be passed as well, the column name should be "label"
                All columns should be complete filenames. 
                The class can take in multiple inputs if columns are named "input-1", "input-2"...
                The class can take in multiple labels if columns are named "label-1", "label-2"...
            n_samples_per_input (int): number of samples per input to be generated each epoch. 
                Samples are randomly extracted according to the label location when "label" is present,
                the smallest input is used otherwise. 
            crop_len (float): length (and width) of the crop to be extracted from training limits.
            ignore_index (int): integer value to be assigned to nan pixels
            transforms (callable, optional): Optional transform to be applied
                on a sample.
            norms (dictionary of callable): Albumentations normalizations for each input used. 
            seed (integer): Seed for random number generator
        """
        self.df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

        self.n_samples_per_input = n_samples_per_input
        self.crop_len = crop_len

        self.transforms = transforms
        self.norms = norms

        self.ignore_index = ignore_index

        # save sample information in a dictionary
        self.samples_dict = {}
        self.samples_dict_full = {}

        # save source/column names
        self.source_names = self.df.columns.tolist()

        # create a BitGenerator for numpy sampling
        self.rs = RandomState(MT19937(SeedSequence(seed)))

        # get the xarray rasters for all data and save as dictionary
        self.dict_xar = self.df.applymap(rioxarray.open_rasterio, masked=True).to_dict(orient='index')

        # prepare the first collection of training samples
        #self.prepare_samples()
    
    def prepare_samples(self):
        """Creates cropped samples to be used for training and validation.
        """

        # reset samples
        self.samples_dict = {}

        # get the names of columns that will be used to extract centroids for 
        # sample generation
        if any('label' in source for source in self.source_names):
            bounds_from = [source for source in self.source_names if 'label' in source]
        else:
            bounds_from = [source for source in self.source_names if 'input' in source]
        
        # TODO: something better than looping?
        for key, val in self.dict_xar.items():
            
            # get smallest bounding box
            min_b_box = np.inf

            for source in bounds_from:
                area = (val[source].x.size)*(val[source].y.size)

                if area < min_b_box:
                    min_b_box = area

                    minx = val[source].x.min()
                    maxx = val[source].x.max()
               
                    miny = val[source].y.min()
                    maxy = val[source].y.max()

                    minbox = val[source]

            # TODO: improve speed, check for possible hangs
            n_samples = 0
            x_samp = []
            y_samp = []
            
            for attempts in range(self.n_samples_per_input*100):

                if n_samples < self.n_samples_per_input:
                    # randomly select locations
                    east = self.rs.uniform(low=minx+self.crop_len/2,
                                            high=maxx-self.crop_len/2,
                                            size=1)

                    north = self.rs.uniform(low=miny+self.crop_len/2,
                                            high=maxy-self.crop_len/2,
                                            size=1)

                    # clip "main" raster for this sample:
                    mask_x = (minbox.x >= east-self.crop_len/2) & (minbox.x <= east+self.crop_len/2)
                    mask_y = (minbox.y >= north-self.crop_len/2) & (minbox.y <= north+self.crop_len/2)

                    # keep if at least 50% of pixels are valid
                    samp = minbox[dict(x=mask_x, y=mask_y)]
                    if samp.isnull().sum().item()/samp[0].values.ravel().shape[0] < .30:
                        x_samp.append(east)
                        y_samp.append(north)
                        n_samples += 1
                
                else:
                    break
                
                if attempts == (self.n_samples_per_input*100)-1:
                    print(f'---Warning! I could only find {n_samples} samples for one of the inputs')
           
            # TODO: something better than looping?
            # clip rasters:
            for east, north in zip(x_samp, y_samp):
                # dictionary to save sample information
                sample_dict = {} 
                for source_name, source in val.items():
                    # use masks
                    mask_x = (source.x >= east-self.crop_len/2) & (source.x <= east+self.crop_len/2)
                    mask_y = (source.y >= north-self.crop_len/2) & (source.y <= north+self.crop_len/2)

                    # save crop
                    sample_dict[source_name] = source[dict(x=mask_x, y=mask_y)]

                    if source_name.startswith('label') and sample_dict[source_name].isnull().any().item():
                        nanmap = sample_dict[source_name].isnull().data
                        for idx, mask in enumerate(nanmap):
                            sample_dict[source_name].data[idx, mask] = self.ignore_index

                    elif sample_dict[source_name].isnull().any().item():
                        nanmap = sample_dict[source_name].isnull().data
                        replace_means = sample_dict[source_name].mean(dim=['x','y']).data

                        # print(f'---Warning, I found {sample_dict[source_name].isnull().sum().item()} null pixels', end=' ')
                        # print(f'in the {source_name} for sample number {len(self.samples_dict)}.', end=' ')
                        # print('They will be replaced with the means for each channel')
                        # print(replace_means)

                        for idx, bmean in enumerate(replace_means):
                            sample_dict[source_name].data[idx, nanmap[idx]] = bmean

                # save sample information in the full sample_dict
                self.samples_dict[len(self.samples_dict)] = sample_dict.copy()

    def set_samples_dict(self, new_samples_dict):
        """
        Update the current self.samples_dict with a new dictionary.
        Useful to update self.sampled_dict with the full overlapping
        dictionary and still make use of the __getitem__ method.
        Args:
            new_samples_dict (dict in the same format as self.samples_dict): 
                A new dictionary to replace the current self.samples_dict.
        """
        self.samples_dict = new_samples_dict

    def __len__(self):
        return len(self.samples_dict)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dict = self.samples_dict[idx]

        sample = {}
        for key, val in sample_dict.items():
            
            sample[key] = val.values

            if self.transforms:

                # albumentations uses channel last
                sample[key] = np.swapaxes(val, 0, -1)

                # albumentations transform needs an 'image' that we discard afterwards
                sample = self.transforms(image=np.zeros((1, 1)),
                                        **sample)
                del sample['image']

                # reorder array
                # albumentations uses channel last
                sample[key] = np.swapaxes(val, 0, -1)

            if key in self.norms.keys():
                # convert to tensor and normalize the data
                sample[key] = self.norms[key](torch.from_numpy(sample[key]))
            else:
                if 'label' in key:
                    # rasters have bands, y should not have a dimension for 
                    # the loss (usually)
                    sample[key] = torch.from_numpy(sample[key]).squeeze(dim=0).long()
                else:
                    sample[key] = torch.from_numpy(sample[key])

        return sample

#############################################
#############################################


class GeoDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):

        if stage == 'fit':

            train_df = self.hparams.df.loc[self.hparams.df['dset'] == 'train'].copy()
            train_df = train_df.drop(['dset'], axis=1)

            val_df = self.hparams.df.loc[self.hparams.df['dset'] == 'val'].copy()
            val_df = val_df.drop(['dset'], axis=1)

            self.train_ds = RasterDataset(df=train_df,
                                    n_samples_per_input=self.hparams.n_samples_per_input,
                                    crop_len=self.hparams.crop_len,
                                    ignore_index=self.hparams.ignore_index,
                                    transforms=self.hparams.transforms,
                                    norms=self.hparams.norms, 
                                    seed=self.hparams.seed)

            self.val_ds = RasterDataset( df=val_df,
                                    n_samples_per_input=self.hparams.n_samples_per_input,
                                    crop_len=self.hparams.crop_len,
                                    ignore_index=self.hparams.ignore_index,
                                    transforms=self.hparams.transforms,
                                    norms=self.hparams.norms, 
                                    seed=self.hparams.seed)

        print(f'{stage} setup complete.')

    def train_dataloader(self):
        print('\nSetting train dataloader')
        self.train_ds.prepare_samples()
        print(f'The data are instance of {type(self.train_ds)}')
        print('A sample of train: ')
        for key, val in self.train_ds[0].items():
            print(f'{key} has shape {val.shape}')

        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        print('\nSetting val dataloader')
        self.val_ds.prepare_samples()
        print(f'The data are instance of {type(self.val_ds)}')
        print('A sample of validation: ')
        for key, val in self.val_ds[0].items():
            print(f'{key} has shape {val.shape}')

        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size)


