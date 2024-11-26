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
from models_dual_mid import DeepLabLikeMultiStreamDict

import time
from pathlib import Path
import rioxarray
import seaborn as sns
import xarray as xr
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, jaccard_score)
from torch import nn
from torchvision.transforms import Normalize

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
    norms['input-1'] = Normalize(mean, std)
    norms['input-2'] = Normalize(mean, std)

    pl.seed_everything(seed, workers=True)

    #########################################################
    for dir_idx, dir_out in enumerate(dir_outs):
        fname_csv = fname_csvs[dir_idx]
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
        torch.save(model.state_dict(), os.path.join(dir_out, "model.pth"))
        
        # run some examples with the trained model
        model.eval()

        for dataload, dset in zip([dm.train_dataloader(), dm.val_dataloader()], ['train', 'validation']):
            fig_counter = 0

            for sample in dataload:
                imgs_b = [sample['input-1'], sample['input-2']]
                tar_b = sample['label']
                
                res_b = model(imgs_b)
                
                for img_center_1, img_center_2, res, tar in zip(imgs_b[0], imgs_b[1], res_b, tar_b):
                    fig, ax = plt.subplots(ncols=4, figsize=(16,4))
                    
                    ax[0].imshow(np.swapaxes(img_center_1.detach().numpy(),0,-1)[:,:,0])
                    ax[1].imshow(np.swapaxes(img_center_2.detach().numpy(),0,-1)[:,:,0])
                    ax[2].imshow(np.swapaxes(res.detach().numpy().argmax(0),0,-1), vmin=0, vmax=num_classes-1)
                    ax[-1].imshow(np.swapaxes(tar.numpy(),0,-1), vmin=0, vmax=num_classes-1)

                    ax[0].set_title('input-1')
                    ax[1].set_title('input-2')
                    ax[2].set_title('prediction')
                    ax[3].set_title('label')

                    fig.suptitle(dset)
                    fig.tight_layout()
                    
                    plt.savefig(os.path.join(dir_out, f'{dset}-{fig_counter}'))
                    fig_counter += 1

                    plt.close('all')

                    if dset == 'train':
                        break
                if fig_counter >= 32:
                    break
        # =========================== TEST ===========================================================
        metrics_path = os.path.normpath(config['io']['metrics_path'])
        metrics_path_1 = os.path.normpath(config['io']['metrics_path_1'])
        metrics_path_2 = os.path.normpath(config['io']['metrics_path_2'])

        test_rasters_1_all = [os.path.normpath(f) for f in config['test']['test_rasters_1'].split('\n')]
        test_rasters_2_all = [os.path.normpath(f) for f in config['test']['test_rasters_2'].split('\n')]
        test_label_rasters_all = [os.path.normpath(f) for f in config['test']['test_label_rasters'].split('\n')]

        # run on test rasters:
        softmax = nn.Softmax(0)

        # Assuming each test split only has two scenes (plus two for alos2 only, two for s1 only, and a null separator)
        test_rasters_1 = test_rasters_1_all[dir_idx*7:dir_idx*7+7]
        test_rasters_2 = test_rasters_2_all[dir_idx*7:dir_idx*7+7]
        test_label_rasters = test_label_rasters_all[dir_idx*7:dir_idx*7+7]

        max_len = max(len(test_rasters_1), len(test_rasters_2))

        for idx in range(max_len):

            test_raster_1 = test_rasters_1[idx] if idx < len(test_rasters_1) and test_rasters_1[idx] != 'null' else None
            test_raster_2 = test_rasters_2[idx] if idx < len(test_rasters_2) and test_rasters_2[idx] != 'null' else None

            if test_raster_1 == None and test_raster_2 == None:
                print('Switching to single input classifier...\n')
                model.update_classifier_single()
                continue

            if test_raster_1 and test_raster_2:
                input_type = 0
                print(f'Using rasters {test_raster_1} and {test_raster_2}...', end=' ')
                rasters = [rioxarray.open_rasterio(test_raster_1, masked=True),
                    rioxarray.open_rasterio(test_raster_2, masked=True)]
            elif test_raster_1:
                input_type = 1
                print(f'Using raster {test_raster_1} only...', end=' ')
                rasters = [rioxarray.open_rasterio(test_raster_1, masked=True)]
            else:
                input_type = 2
                print(f'Using raster {test_raster_2} only...', end=' ')
                rasters = [rioxarray.open_rasterio(test_raster_2, masked=True)]

            start_time = time.perf_counter()
            x = [torch.from_numpy(raster.values).unsqueeze(dim=0) for raster in rasters]

            # get input mask 
            mask = np.isnan(rasters[0].values).any(axis=0)
            if len(rasters) > 1:
                mask |= np.isnan(rasters[1].values).any(axis=0)

            with torch.no_grad():
                x = [norms[f'input-{i+1}'](t) for i, t in enumerate(x)]
                x = [torch.nan_to_num(t) for t in x]
                res = model(x, input_type=input_type)
                # compute probabilities (instead of scores):
                res = softmax(torch.squeeze(res,0))
            
            end_time = time.perf_counter()
            print(f'{(end_time-start_time)/60:.2f} minutes for model prediction...', end=' ')
            start_time = time.perf_counter()

            # cast results to numpy
            res = res.detach().numpy()

            # mark nan vals
            for band in res:
                band[mask] = np.nan

            raster = rasters[0]

            # use raster information to populate output:
            xr_res = xr.DataArray(res, 
                                [('band', np.arange(1, res.shape[0]+1)),
                                ('y', raster.y.values),
                                ('x', raster.x.values)])
            
            xr_res['spatial_ref']=raster.spatial_ref                              
            xr_res.attrs=raster.attrs
            xr_res.attrs = {k: v for k, v in xr_res.attrs.items() if k in ['scale_factor', 'add_offset']}
            
            # write to file
            if test_raster_1 and test_raster_2:
                out_fname = os.path.join(dir_out, f'pred-{Path(test_raster_1).stem}_and_{Path(test_raster_2).stem}.tif')
            elif test_raster_1:
                out_fname = os.path.join(dir_out, f'pred-{Path(test_raster_1).stem}.tif')
            elif test_raster_2:
                out_fname = os.path.join(dir_out, f'pred-{Path(test_raster_2).stem}.tif')
            else:
                raise ValueError("No valid raster provided for output file name.")
            if os.path.isfile(out_fname):
                os.remove(out_fname)
            xr_res.rio.to_raster(out_fname, dtype="float32")
            
            # write the class
            y_pred_class = res.argmax(0)
            # 241 is the no data value for uint8
            nodata = 241
            y_pred_class[mask] = nodata
            y_pred_class = np.expand_dims(y_pred_class, 0)

            xr_res = xr.DataArray(y_pred_class, 
                                [('band', [1]),
                                ('y', raster.y.values),
                                ('x', raster.x.values)])
            
            xr_res['spatial_ref']=raster.spatial_ref                              
            xr_res.attrs=raster.attrs
            xr_res.attrs = {k: v for k, v in xr_res.attrs.items() if k in ['scale_factor', 'add_offset']}

            xr_res.rio.write_nodata(nodata, inplace=True)
            
            if test_raster_1 and test_raster_2:
                out_fname = os.path.join(dir_out, f'class-{Path(test_raster_1).stem}_and_{Path(test_raster_2).stem}.tif')
            elif test_raster_1:
                out_fname = os.path.join(dir_out, f'class-{Path(test_raster_1).stem}.tif')
            elif test_raster_2:
                out_fname = os.path.join(dir_out, f'class-{Path(test_raster_2).stem}.tif')
            else:
                raise ValueError("No valid raster provided for output file name.")
            if os.path.isfile(out_fname):
                os.remove(out_fname)
            xr_res.rio.to_raster(out_fname, dtype="uint8")

            # compute metrics if the labels are available
            if 0 <= idx < len(test_label_rasters):
                raster_y = rioxarray.open_rasterio(test_label_rasters[idx], masked=True)

                y_true = np.squeeze(raster_y.values, 0)
                y_true[y_true==ignore_index]=np.nan
                mask_y = np.isnan(y_true)

                y_pred_class = np.squeeze(y_pred_class, 0)

                # write correctly labeled pixels
                correct = np.array(y_true==y_pred_class).astype('uint8')
                correct[mask_y] = nodata

                xr_res = xr.DataArray(np.expand_dims(correct, 0), 
                                    [('band', [1]),
                                    ('y', raster.y.values),
                                    ('x', raster.x.values)])
                
                xr_res['spatial_ref']=raster.spatial_ref                              
                xr_res.attrs=raster.attrs
                xr_res.attrs = {k: v for k, v in xr_res.attrs.items() if k in ['scale_factor', 'add_offset']}

                xr_res.rio.write_nodata(nodata, inplace=True)

                if test_raster_1 and test_raster_2:
                    out_fname = os.path.join(dir_out, f'correct-prim-{Path(test_raster_1).stem}_and_{Path(test_raster_2).stem}-vs-{Path(test_label_rasters[idx]).stem}.tif')
                elif test_raster_1:
                    out_fname = os.path.join(dir_out, f'correct-prim-{Path(test_raster_1).stem}-vs-{Path(test_label_rasters[idx]).stem}.tif')
                elif test_raster_2:
                    out_fname = os.path.join(dir_out, f'correct-prim-{Path(test_raster_2).stem}-vs-{Path(test_label_rasters[idx]).stem}.tif')
                else:
                    raise ValueError("No valid raster provided for output file name.")
                if os.path.isfile(out_fname):
                    os.remove(out_fname)
                xr_res.rio.to_raster(out_fname, dtype="uint8")

                if test_raster_1 and test_raster_2:
                    metrics_txt = metrics_path
                elif test_raster_1:
                    metrics_txt = metrics_path_1
                elif test_raster_2:
                    metrics_txt = metrics_path_2
                else:
                    raise ValueError("No valid raster provided for output file name.")

                with open(metrics_txt, 'a', encoding='utf-8') as outfile:
                    if test_raster_1 and test_raster_2:
                        outfile.write(f'{Path(test_raster_1).stem}_and_{Path(test_raster_2).stem} vs {Path(test_label_rasters[idx]).stem} performance \n')
                    elif test_raster_1:
                        outfile.write(f'{Path(test_raster_1).stem} vs {Path(test_label_rasters[idx]).stem} performance \n')
                    elif test_raster_2:
                        outfile.write(f'{Path(test_raster_2).stem} vs {Path(test_label_rasters[idx]).stem} performance \n')
                    else:
                        raise ValueError("No valid raster provided for output file name.")
                    outfile.write(classification_report(y_true[~np.logical_or(mask, mask_y)].ravel(), y_pred_class[~np.logical_or(mask, mask_y)].ravel()))
                    outfile.write('\n')
                    outfile.write(f'Jaccard Index: \n')
                    for avg in ['micro', 'macro', 'weighted']:
                        iou = jaccard_score(y_true[~np.logical_or(mask, mask_y)].ravel(), 
                                            y_pred_class[~np.logical_or(mask, mask_y)].ravel(), average=avg)
                        outfile.write(f'{avg}: {iou:.2f} \n')

                    cm = confusion_matrix(y_true[~np.logical_or(mask, mask_y)].ravel(), 
                                            y_pred_class[~np.logical_or(mask, mask_y)].ravel(), 
                                            normalize='true')
                    outfile.write('\n')
                    outfile.write(f'Confusion Matrix: \n')
                    for row in cm:
                        for col in row:
                            outfile.write(f'      {col:.2f}')
                        outfile.write('\n')
                    outfile.write('\n\n')

                    # save pdf 
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_predictions(y_true[~np.logical_or(mask, mask_y)].ravel(), 
                                                            y_pred_class[~np.logical_or(mask, mask_y)].ravel(), 
                                                            normalize='true', 
                                                            values_format = '.2f',
                                                            ax=ax)
                    

                    if test_raster_1 and test_raster_2:
                        fig.savefig(os.path.join(dir_out, f'confusion_matrix-{Path(test_raster_1).stem}_and_{Path(test_raster_2).stem}_vs_{Path(test_label_rasters[idx]).stem}.pdf'))
                    elif test_raster_1:
                        fig.savefig(os.path.join(dir_out, f'confusion_matrix-{Path(test_raster_1).stem}_vs_{Path(test_label_rasters[idx]).stem}.pdf'))
                    elif test_raster_2:
                        fig.savefig(os.path.join(dir_out, f'confusion_matrix-{Path(test_raster_2).stem}_vs_{Path(test_label_rasters[idx]).stem}.pdf'))
                    else:
                        raise ValueError("No valid raster provided for output file name.")
            
                y_true = None
                correct = None

                plt.close('all')

            # this uses a lot of memory, delete some stuff:
            raster = None
            x = None
            res = None
            mask = None

            end_time = time.perf_counter()
            print(f'{(end_time-start_time)/60:.2f} minutes for writing files and metrics')

        # ============================================================================================
        
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

