from typing import List

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torch.nn import ModuleList
from torch.nn import functional as F
from torchmetrics import JaccardIndex
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from losses import FocalLossMod


###############################################
# model that receives a dictionary as input
###############################################
class DeepLabLikeMultiStreamDict(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.pretrained == True:
            weights=ResNet18_Weights.IMAGENET1K_V1
        else:
            weights=None

        self.streams = ModuleList()
        return_layers = {"layer3": "out"}
        
        backbone_1 = resnet18(weights=weights)
        backbone_1.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.streams.append(IntermediateLayerGetter(backbone_1, return_layers=return_layers))

        backbone_2 = resnet18(weights=weights)
        backbone_2.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.streams.append(IntermediateLayerGetter(backbone_2, return_layers=return_layers))

        # create the head:
        self.classifier_single = DeepLabHead(256, self.hparams.num_classes)
        self.classifier_double = DeepLabHead(256 * 2, self.hparams.num_classes)

        # DeepLabHead has one global average pooling that messes things up 
        # in case we want to change the input size (full raster prediction)
        avgpool_replacer = nn.AvgPool2d(2,2)
        for classifier in [self.classifier_single, self.classifier_double]:
            if isinstance(classifier[0].convs[-1][0], nn.AdaptiveAvgPool2d):
                classifier[0].convs[-1][0] = avgpool_replacer
            else:
                print('Check the model! Is there an AdaptiveAvgPool2d somewhere?')

        # metrics
        # use the num_classes+1 to ignore last index
        # torchmetrics Jaccard Index does not work when average='weighted' & ignore_index is present
        # use macro as underperforming classes will have a higher influence in the final value
        self.jaccard = JaccardIndex(num_classes=self.hparams.num_classes+1, 
                                    average='macro', 
                                    ignore_index = self.hparams.ignore_index)

        # loss
        if self.hparams.loss == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss(weight = self.hparams.class_weights, 
                                               label_smoothing = 0.0, 
                                               ignore_index = self.hparams.ignore_index)
        elif self.hparams.loss == 'focal':
            self.loss_fn = FocalLossMod(gamma=self.hparams.gamma, 
                                        alpha=self.hparams.alpha, 
                                        reduction='mean', 
                                        ignore_index = self.hparams.ignore_index)
            
    def update_classifier_single(self):
        # Copy weights from classifier_double to classifier_single
        classifier_double_state = self.classifier_double.state_dict()
        new_state_dict = self.classifier_single.state_dict()

        for key in new_state_dict:
            if 'weight' in key and len(classifier_double_state[key].shape) > 1:
                # If the layer has mismatched input channels (512 in double vs 256 in single)
                if classifier_double_state[key].shape[1] == 512 and new_state_dict[key].shape[1] == 256:
                    new_state_dict[key] = (
                        classifier_double_state[key][:, :256, :, :] +  # First 256 input channels
                        classifier_double_state[key][:, 256:, :, :]  # Second 256 input channels
                    ) / 2
                else:
                    # Copy other weights directly if the dimensions match
                    new_state_dict[key] = classifier_double_state[key].clone()
            else:
                # Copy all other parameters without modification
                new_state_dict[key] = classifier_double_state[key].clone()

        # Load the new state_dict into classifier_single
        self.classifier_single.load_state_dict(new_state_dict)
        
    def forward(self, x:List[Tensor], input_type:int = 0) -> Tensor:

        
        input_shape = x[0].shape[-2:]
        
        if input_type == 0:
            features = [stream(x[i])['out'] for i, stream in enumerate(self.streams)]
            features = torch.cat(features, dim=1)
            y_hat = self.classifier_double(features)
        else:
            if input_type == 1:
                features = self.streams[0](x[0])['out']
            else:
                features = self.streams[1](x[0])['out']
            y_hat = self.classifier_single(features)

        y_hat = F.interpolate(y_hat, size=input_shape, mode="bilinear", align_corners=False)
        
        return y_hat

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)   
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                        min_lr=1e-8,
                                                                        patience=self.hparams.reduce_lr_patience, 
                                                                        verbose=True)         
            return optimizer 

    def training_step(self, batch, batch_idx):

        if 'input-1' in batch and 'input-2' in batch:
            x = [batch['input-1'], batch['input-2']]
            input_type = 0
        elif 'input-1' in batch:
            x = [batch['input-1']]
            input_type = 1
        else:
            x = [batch['input-2']]
            input_type = 2
        y = batch['label']

        y_pred = self.forward(x, input_type=input_type)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)

        ###############################################
        # metrics
        ###############################################
        # IoU/Jaccard index
        iou = self.jaccard(torch.argmax(y_pred, axis=1), y)
        self.log('train_IoU', iou, on_epoch=True)

        # F1 
        f1 = torchmetrics.functional.f1_score(torch.argmax(y_pred, axis=1).ravel(), y.ravel(),
                                        average='macro',
                                        num_classes=self.hparams.num_classes+1,
                                        ignore_index=self.hparams.ignore_index)

        self.log('train_f1', f1, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if 'input-1' in batch and 'input-2' in batch:
            x = [batch['input-1'], batch['input-2']]
            input_type = 0
        elif 'input-1' in batch:
            x = [batch['input-1']]
            input_type = 1
        else:
            x = [batch['input-2']]
            input_type = 2
        y = batch['label']

        y_pred = self.forward(x, input_type=input_type)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_epoch=True)

        ###############################################
        # metrics
        ###############################################
        # IoU/Jaccard index
        iou = self.jaccard(torch.argmax(y_pred, axis=1), y)
        self.log('val_IoU', iou, on_epoch=True)

        # F1
        f1 = torchmetrics.functional.f1_score(torch.argmax(y_pred, axis=1).ravel(), y.ravel(),
                                        average='macro',
                                        num_classes=self.hparams.num_classes+1,
                                        ignore_index=self.hparams.ignore_index)

        self.log('val_f1', f1, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        sch = self.scheduler

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])                       
