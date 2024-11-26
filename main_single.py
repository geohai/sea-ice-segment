import argparse
import configparser
import os
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torchvision.transforms import Normalize

from datasets import GeoDataModule
from models_single import DeepLabLikeMultiStreamDict


def main(config):

    dir_outs = [os.path.normpath(f) for f in config['io']['dir_out'].split('\n')]
    fname_csvs = [os.path.normpath(f) for f in config['io']['fname_csv'].split('\n')]

    num_streams = int(config['model']['num_streams'])
    num_classes = int(config['model']['num_classes'])
    pretrained = config['model']['pretrained'] == 'True'
    
    loss = config['loss']['loss']
    if loss == 'focal':
        gamma = float(config['loss']['gamma'])
        alpha = float(config['loss']['alpha'])
    else:
        gamma = None
        alpha = None

    n_samples_per_input = int(config['datamodule']['n_samples_per_input'])
    crop_len = float(config['datamodule']['crop_len'])
    seed = int(config['datamodule']['seed'])
    if 'class_weights' in config['datamodule'].keys():
        class_weights = [float(val) for val in config['datamodule']['class_weights'].split(',')]
        class_weights = torch.Tensor(class_weights)
    else:
        class_weights = None

    mean = [float(val) for val in config['datamodule']['mean'].split(',')]
    std = [float(val) for val in config['datamodule']['std'].split(',')]

    min_epochs = int(config['train']['min_epochs'])
    max_epochs = int(config['train']['max_epochs'])
    patience = int(config['train']['patience'])
    reduce_lr_patience = int(config['train']['reduce_lr_patience'])
    batch_size = int(config['train']['batch_size'])
    lr = float(config['train']['lr'])
    reload_every_n_epochs = int(config['train']['reload_every_n_epochs'])

    fine_tune = config['train']['fine_tune'] == 'True'
    ignore_index = int(config['train']['ignore_index'])

    transforms = None
    norms = {}
    norms['input'] = Normalize(mean, std)

    pl.seed_everything(seed, workers=True)

    #########################################################
    for idx, dir_out in enumerate(dir_outs):
        fname_csv = fname_csvs[idx]
        df = pd.read_csv(fname_csv)

        # Windows need different distributed backend
        # this will be depecrated in pytorch lightning in version 1.8
        # and a full Strategy object will have to be set up
        if os.name == 'nt':
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

        ###########################################################
        # Set the pl.DataModule to be used in experiments
        ###########################################################

        # create datamodule:
        dm = GeoDataModule(df=df,
                        n_samples_per_input=n_samples_per_input,
                        crop_len=crop_len,
                        transforms=transforms,
                        norms=norms,
                        ignore_index=ignore_index,
                        seed=seed, 
                        batch_size=batch_size, 
                        )

        ###########################################################
        # create models
        ###########################################################

        model = DeepLabLikeMultiStreamDict(num_streams=num_streams, 
                                        num_classes=num_classes,
                                        pretrained=pretrained,
                                        loss=loss,
                                        gamma=gamma,
                                        alpha=alpha,
                                        class_weights = class_weights,
                                        lr=lr, 
                                        reduce_lr_patience=reduce_lr_patience, 
                                        ignore_index=ignore_index)

        if fine_tune:
            if os.path.isfile(os.path.normpath(config['io']['fname_model'])):
                model.load_from_checkpoint(os.path.normpath(config['io']['fname_model']))
            else:
                print(f"{os.path.normpath(config['io']['fname_model'])} is not a valid model")
                return

        ###########################################################
        # run experiment
        ###########################################################

        # callbacks:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=patience,
            verbose=False,
            mode='min'
        )

        best_weights = ModelCheckpoint(dirpath=dir_out,
                                        filename=f'best_weights',
                                        save_top_k=1,
                                        verbose=False,
                                        monitor='val_loss',
                                        mode='min'
                                        )
        

        # loggers:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=dir_out)
        cvs_logger = pl_loggers.CSVLogger(save_dir=dir_out)

        # make sure there are no old files there
        if os.path.isfile(os.path.join(dir_out, 'best_weights.ckpt')):
            os.remove(os.path.join(dir_out, 'best_weights.ckpt'))

        trainer = pl.Trainer(gpus=1,
                            default_root_dir=dir_out,
                            gradient_clip_val=1.0,   # clip large gradients
                            log_every_n_steps=1,
                            min_epochs=min_epochs,
                            callbacks=[early_stopping, best_weights],
                            logger=[tb_logger, cvs_logger],
                            strategy="ddp",
                            accelerator="gpu",
                            max_epochs=max_epochs,
                            reload_dataloaders_every_n_epochs=reload_every_n_epochs,
                            )
        
        trainer.fit(model, dm)

        # TODO: maybe remove the quick check
        # pytorch lightning "does not consider" script continues after .fit, 
        # the following code will likely be ran once for each gpu available
        # https://stackoverflow.com/questions/66261729/pytorch-lightning-duplicates-main-script-in-ddp-mode
        if model.global_rank != 0:
            sys.exit(0)

        # make sure model has the best weights and not the ones for the last epoch
        if os.path.isfile(os.path.join(dir_out, 'best_weights.ckpt')):
            model.load_from_checkpoint(os.path.join(dir_out, 'best_weights.ckpt'))

        # save for use in production environment
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(dir_out, "model.pt"))
        
        # run some examples with the trained model
        model.eval()

        for dataload, dset in zip([dm.train_dataloader(), dm.val_dataloader()], ['train', 'validation']):
            fig_counter = 0

            for sample in dataload:
                imgs_b = [sample['input']]
                tar_b = sample['label']
                
                res_b = model(imgs_b)
                
                for img_center, res, tar in zip(imgs_b[0], res_b, tar_b):
                    fig, ax = plt.subplots(ncols=3, figsize=(12,4))
                    
                    ax[0].imshow(np.swapaxes(img_center.detach().numpy(),0,-1)[:,:,0])
                    ax[1].imshow(np.swapaxes(res.detach().numpy().argmax(0),0,-1), vmin=0, vmax=num_classes-1)
                    ax[-1].imshow(np.swapaxes(tar.numpy(),0,-1), vmin=0, vmax=num_classes-1)

                    ax[0].set_title('input')
                    ax[1].set_title('prediction')
                    ax[2].set_title('label')

                    fig.suptitle(dset)
                    fig.tight_layout()
                    
                    plt.savefig(os.path.join(dir_out, f'{dset}-{fig_counter}'))
                    fig_counter += 1

                    plt.close('all')

                    if dset == 'train':
                        break
                if fig_counter >= 32:
                    break
        
        # save configuration file:
        with open(os.path.join(dir_out, 'train.cfg'), 'w') as out_file:
            config.write(out_file)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='config_main.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        main(config)
    
    else:
        print('Please provide a valid configuration file.')

