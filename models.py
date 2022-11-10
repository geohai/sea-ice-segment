from typing import List

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torch.nn import ModuleList
from torch.nn import functional as F
from torchmetrics import JaccardIndex
#from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from losses import FocalLossMod

# class DeepLabLikeMultiStream(pl.LightningModule):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()

#         if self.hparams.pretrained == True:
#             weights=ResNet18_Weights.IMAGENET1K_V2
#         else:
#             weights=None

#         # create multiple streams
#         streams = []
#         return_layers = {"layer4": "out"}
#         for _ in range(self.hparams.num_streams):
#             backbone = resnet18(replace_stride_with_dilation=[False, True, True], 
#                                 weights=weights)
#             backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
#             streams.append(backbone)
        
#         self.streams=ModuleList(streams)

#         # create the head:
#         #self.classifier = DeepLabHead(2048*self.hparams.num_streams, self.hparams.num_classes)
#         self.classifier = DeepLabHead(512*self.hparams.num_streams, self.hparams.num_classes)

#         # DeepLabHead has one global average pooling that messes things up 
#         # in case we want to change the input size (full raster prediction)
#         avgpool_replacer = nn.AvgPool2d(32,32)
#         if isinstance(self.classifier[0].convs[-1][0], nn.AdaptiveAvgPool2d):
#             self.classifier[0].convs[-1][0] = avgpool_replacer
#         else:
#             print('Check the model! Is there an AdaptiveAvgPool2d somewhere?')

#         # metrics
#         self.jaccard = JaccardIndex(num_classes=self.hparams.num_classes, average='weighted')
        
#     def forward(self, x:List[Tensor]) -> Tensor:

#         input_shape = x[0].shape[-2:]
        
#         _features = []
#         for stream, xin in zip(self.streams, x):
#             _features.append(stream(xin)['out'])
#         features = torch.cat(_features, dim=1)

#         y_hat = self.classifier(features)        
#         y_hat = F.interpolate(y_hat, size=input_shape, mode="bilinear", align_corners=False)
        
#         return y_hat

#     def configure_optimizers(self):
#             optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)   
#             self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                                                         min_lr=1e-5,
#                                                                         patience=5, 
#                                                                         verbose=True)         
#             return optimizer 

#     def training_step(self, batch, batch_idx):
#         x,y = batch

#         y_pred = self.forward(x)
#         loss = self.hparams.loss_fn(y_pred, y)
#         self.log('train_loss', loss, on_epoch=True)

#         ###############################################
#         # metrics
#         ###############################################
#         # IoU/Jaccard index
#         iou = self.jaccard(y_pred, y)
#         self.log('train_IoU', iou, on_epoch=True)

#         # F1
#         y_true = y.contiguous().view(-1, 1)
#         # N,C,H,W => N,C,H*W
#         y_pred = y_pred.contiguous().view(y_pred.size(0), y_pred.size(1), -1)
#         # N,C,H*W => N,H*W,C
#         y_pred = y_pred.transpose(1, 2)
#         # N,H*W,C => N*H*W,C
#         y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
#         f1 = torchmetrics.functional.f1_score(y_pred, y_true,
#                                         average='weighted',
#                                         num_classes=self.hparams.num_classes)
#         self.log('train_f1', f1, on_epoch=True)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         x,y = batch

#         y_pred = self.forward(x)
#         loss = self.hparams.loss_fn(y_pred, y)
#         self.log('val_loss', loss, on_epoch=True)

#         ###############################################
#         # metrics
#         ###############################################
#         # IoU/Jaccard index
#         iou = self.jaccard(y_pred, y)
#         self.log('val_IoU', iou, on_epoch=True)

#         # F1
#         y_true = y.contiguous().view(-1, 1)
#         # N,C,H,W => N,C,H*W
#         y_pred = y_pred.contiguous().view(y_pred.size(0), y_pred.size(1), -1)
#         # N,C,H*W => N,H*W,C
#         y_pred = y_pred.transpose(1, 2)
#         # N,H*W,C => N*H*W,C
#         y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
#         f1 = torchmetrics.functional.f1_score(y_pred, y_true,
#                                         average='weighted',
#                                         num_classes=self.hparams.num_classes)
#         self.log('val_f1', f1, on_epoch=True)

#         return loss

#     def training_epoch_end(self, outputs):
#         sch = self.scheduler

#         # If the selected scheduler is a ReduceLROnPlateau scheduler.
#         if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             sch.step(self.trainer.callback_metrics["val_loss"])                       

# ###############################################
# # Multi loss model
# ###############################################
# class DeepLabLikeMultiStreamMultiLoss(pl.LightningModule):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()

#         if self.hparams.pretrained == True:
#             weights=ResNet18_Weights.IMAGENET1K_V2
#         else:
#             weights=None

#         # create multiple streams
#         streams = []
#         heads = []
#         return_layers = {"layer4": "out"}
#         for _ in range(self.hparams.num_streams):
#             backbone = resnet18(replace_stride_with_dilation=[False, True, True], 
#                                 weights=weights)
#             backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
#             streams.append(backbone)

#             #heads.append(DeepLabHead(2048, self.hparams.num_classes))
#             heads.append(DeepLabHead(512, self.hparams.num_classes))
        
#         self.streams=ModuleList(streams)
#         self.heads=ModuleList(heads)

#         # create the main head:
#         #self.classifier = DeepLabHead(2048*self.hparams.num_streams, self.hparams.num_classes)
#         self.classifier = DeepLabHead(512*self.hparams.num_streams, self.hparams.num_classes)

#         # DeepLabHead has one global average pooling that messes things up 
#         # in case we want to change the input size (full raster prediction)
#         avgpool_replacer = nn.AvgPool2d(32,32)
#         if isinstance(self.classifier[0].convs[-1][0], nn.AdaptiveAvgPool2d):
#             self.classifier[0].convs[-1][0] = avgpool_replacer
#         else:
#             print('Check the model! Is there an AdaptiveAvgPool2d somewhere?')

#         # metrics
#         self.jaccard = JaccardIndex(num_classes=self.hparams.num_classes, average='weighted')
        
#     def forward(self, x:List[Tensor]) -> Tensor:

#         input_shape = x[0].shape[-2:]

#         _features = []
#         y_hats_secondary=[]

#         for stream, head, xin in zip(self.streams, self.heads, x):
#             # compute the features
#             _feats = stream(xin)['out']
#             # store them for the main classifier
#             _features.append(_feats)

#             # calculate the independent heads
#             _y_hat = head(_feats)
#             _y_hat= F.interpolate(y_hat, size=input_shape, mode="bilinear", align_corners=False)
#             y_hats_secondary.append(_y_hat)

#         # assemble the main features        
#         features = torch.cat(_features, dim=1)

#         # get the main prediction
#         y_hat = self.classifier(features)        
#         y_hat = F.interpolate(y_hat, size=input_shape, mode="bilinear", align_corners=False)
        
#         return y_hat, y_hats_secondary

#     def configure_optimizers(self):
#             optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)   
#             self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                                                         min_lr=1e-5,
#                                                                         patience=5, 
#                                                                         verbose=True)         
#             return optimizer 

#     def training_step(self, batch, batch_idx):
#         x,y = batch

#         y_pred, y_pred_secondary = self.forward(x)
        
#         loss = self.hparams.loss_fn(y_pred, y)
#         loss_secondary = 0
#         for loss_sec, y_pred_sec in zip(self.hparams.loss_secs, y_pred_secondary):
#             loss_secondary+=loss_sec(y_pred_sec, y)

#         loss = self.hparams.alpha*loss + self.hparams.beta*loss_secondary

#         self.log('train_loss', loss, on_epoch=True)

#         ###############################################
#         # metrics
#         ###############################################
#         # IoU/Jaccard index
#         iou = self.jaccard(y_pred, y)
#         self.log('train_IoU', iou, on_epoch=True)

#         # F1
#         y_true = y.contiguous().view(-1, 1)
#         # N,C,H,W => N,C,H*W
#         y_pred = y_pred.contiguous().view(y_pred.size(0), y_pred.size(1), -1)
#         # N,C,H*W => N,H*W,C
#         y_pred = y_pred.transpose(1, 2)
#         # N,H*W,C => N*H*W,C
#         y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
#         f1 = torchmetrics.functional.f1_score(y_pred, y_true,
#                                         average='weighted',
#                                         num_classes=self.hparams.num_classes)
#         self.log('train_f1', f1, on_epoch=True)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         x,y = batch

#         y_pred, y_pred_secondary = self.forward(x)
        
#         loss = self.hparams.loss_fn(y_pred, y)
#         loss_secondary = 0
#         for loss_sec, y_pred_sec in zip(self.hparams.loss_secs, y_pred_secondary):
#             loss_secondary+=loss_sec(y_pred_sec, y)

#         loss = self.hparams.alpha*loss + self.hparams.beta*loss_secondary
        
#         self.log('val_loss', loss, on_epoch=True)

#         ###############################################
#         # metrics
#         ###############################################
#         # IoU/Jaccard index
#         iou = self.jaccard(y_pred, y)
#         self.log('val_IoU', iou, on_epoch=True)

#         # F1
#         y_true = y.contiguous().view(-1, 1)
#         # N,C,H,W => N,C,H*W
#         y_pred = y_pred.contiguous().view(y_pred.size(0), y_pred.size(1), -1)
#         # N,C,H*W => N,H*W,C
#         y_pred = y_pred.transpose(1, 2)
#         # N,H*W,C => N*H*W,C
#         y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
#         f1 = torchmetrics.functional.f1_score(y_pred, y_true,
#                                         average='weighted',
#                                         num_classes=self.hparams.num_classes)
#         self.log('val_f1', f1, on_epoch=True)

#         return loss

#     def training_epoch_end(self, outputs):
#         sch = self.scheduler

#         # If the selected scheduler is a ReduceLROnPlateau scheduler.
#         if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             sch.step(self.trainer.callback_metrics["val_loss"])                       


###############################################
# model that receives a dictionary as input
###############################################
class DeepLabLikeMultiStreamDict(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.pretrained == True:
            #weights=ResNet50_Weights.IMAGENET1K_V2 # 18 does not have v2
            weights=ResNet18_Weights.IMAGENET1K_V1
        else:
            weights=None

        # create multiple streams
        streams = []
        return_layers = {"layer3": "out"}
        for _ in range(self.hparams.num_streams):
            #backbone = resnet50(replace_stride_with_dilation=[False, True, True], weights=weights)
            backbone = resnet18(weights=weights)
            backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
            streams.append(backbone)
        
        self.streams=ModuleList(streams)

        # create the head:
        #self.classifier = DeepLabHead(2048*self.hparams.num_streams, self.hparams.num_classes)
        #self.classifier = DeepLabHead(1024*self.hparams.num_streams, self.hparams.num_classes)
        #self.classifier = DeepLabHead(512*self.hparams.num_streams, self.hparams.num_classes)
        self.classifier = DeepLabHead(256*self.hparams.num_streams, self.hparams.num_classes)

        # DeepLabHead has one global average pooling that messes things up 
        # in case we want to change the input size (full raster prediction)
        avgpool_replacer = nn.AvgPool2d(2,2)
        if isinstance(self.classifier[0].convs[-1][0], nn.AdaptiveAvgPool2d):
            self.classifier[0].convs[-1][0] = avgpool_replacer
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
        
    def forward(self, x:List[Tensor]) -> Tensor:

        
        input_shape = x[0].shape[-2:]
        
        _features = []
        # need enumerate for torch.jit.script 
        for idx, stream in enumerate(self.streams):
            _features.append(stream(x[idx])['out'])
        features = torch.cat(_features, dim=1)

        y_hat = self.classifier(features)        
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

        x = [batch['input']]
        y = batch['label']

        y_pred = self.forward(x)
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

        # # F1 - unfortunately `ignore_index` for torchmetrics F1
        # # works a little differently, so we reshape the data
        # y_true = y.contiguous().view(-1, 1)
        # # N,C,H,W => N,C,H*W
        # y_pred = y_pred.contiguous().view(y_pred.size(0), y_pred.size(1), -1)
        # # N,C,H*W => N,H*W,C
        # y_pred = y_pred.transpose(1, 2)
        # # N,H*W,C => N*H*W,C
        # y_pred = y_pred.contiguous().view(-1, y_pred.size(2))

        # mask = y_true == self.hparams.ignore_index
        # mask = mask.ravel()

        # f1 = torchmetrics.functional.f1_score(y_pred[~mask], y_true[~mask],
        #                                 average='weighted',
        #                                 num_classes=self.hparams.num_classes)

        self.log('train_f1', f1, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = [batch['input']]
        y = batch['label']

        y_pred = self.forward(x)
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

        # # F1 - unfortunately `ignore_index` for torchmetrics F1
        # # works a little differently, so we reshape the data
        # y_true = y.contiguous().view(-1, 1)
        # # N,C,H,W => N,C,H*W
        # y_pred = y_pred.contiguous().view(y_pred.size(0), y_pred.size(1), -1)
        # # N,C,H*W => N,H*W,C
        # y_pred = y_pred.transpose(1, 2)
        # # N,H*W,C => N*H*W,C
        # y_pred = y_pred.contiguous().view(-1, y_pred.size(2))

        # mask = y_true == self.hparams.ignore_index
        # mask = mask.ravel()

        # f1 = torchmetrics.functional.f1_score(y_pred[~mask], y_true[~mask],
        #                                 average='weighted',
        #                                 num_classes=self.hparams.num_classes)

        self.log('val_f1', f1, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        sch = self.scheduler

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])                       
