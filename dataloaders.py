"""
Dataset and dataloaders
"""
import glob
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# modified from
class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path, n_samples):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(f"{os.path.join(folder_path,'original_images')}/*.jpg")
        self.img_files = self.img_files[:n_samples]
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,
                                                 #'val-label-img',
                                                 'label_images_semantic',
                                                 f'{Path(img_path).stem}.png'))#_lab.png'))
        
    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = Image.open(img_path, mode='r', formats=None)
            label =Image.open(mask_path, mode='r', formats=None)
                                                                
            label_arr = np.array(label)
            return transforms.ToTensor()(data), torch.LongTensor(label_arr)

    def __len__(self):
        return len(self.img_files)

class ModDataLoaderSegmentation(DataLoaderSegmentation):
    def __init__(self, folder_path, n_samples, size, resize_to):
        super().__init__(folder_path, n_samples)

        self.size = size
        self.resize_to = resize_to

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = Image.open(img_path, mode='r', formats=None)
        label =Image.open(mask_path, mode='r', formats=None)

        data = data.resize((self.resize_to,self.resize_to))
        label= label.resize((self.resize_to,self.resize_to), resample=Image.Resampling.NEAREST)

        # transform the input
        # these values are not correct as this is from imagenet most likely
        norms = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                     std=(0.229, 0.224, 0.225))
        
        # crop the center of the image:
        crop_center = transforms.CenterCrop(size=self.size)
        # crop a random location:
        crop_random = transforms.RandomCrop(size=self.size)
        
        data_crop = crop_center(data)
        data_crop_mis = crop_random(data)
        label_arr = np.array(crop_center(label))
        
        data_crop = norms(transforms.ToTensor()(data_crop))
        data_crop_mis = norms(transforms.ToTensor()(data_crop_mis))

        # separate by channels:
        data_crop[:,:,:]=data_crop[0,:,:]

        data_crop_mis[:,:,:]=data_crop_mis[1,:,:]


        return [data_crop, data_crop_mis], torch.LongTensor(label_arr)


    def __len__(self):
        return len(self.img_files)
    

class DModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage):
        data_full = ModDataLoaderSegmentation(folder_path=self.hparams.folder_path, 
                                           n_samples=self.hparams.n_training_samples,
                                           size=self.hparams.out_size,
                                           resize_to=self.hparams.resize_to
                                           )
        torch.manual_seed(0)
        self.train, self.val = random_split(data_full,
                            [len(data_full)-self.hparams.n_val_samples,
                             self.hparams.n_val_samples], )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
