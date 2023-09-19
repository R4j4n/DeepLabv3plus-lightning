import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT 
import torch
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset , DataLoader
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models

class ResNet_50(nn.Module):
    def __init__(self, output_layer=None):
        super(ResNet_50, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x
    
class Atrous_Convolution(nn.Module):
    """Compute Atrous/Dilated Convolution.
    """

    def __init__(
            self, input_channels, kernel_size, pad, dilation_rate,
            output_channels=256):
        super(Atrous_Convolution, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size, padding=pad,
                              dilation=dilation_rate, bias=False)

        self.batchnorm = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ASSP(nn.Module):
    """Atrous Spatial Pyramid pooling layer
    """

    def __init__(self, in_channles, out_channles):
        """Atrous Spatial Pyramid pooling layer
        Args:
            in_channles (int): No of input channel for Atrous_Convolution.
            out_channles (int): No of output channel for Atrous_Convolution.
        """
        super(ASSP, self).__init__()
        self.conv_1x1 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=1, pad=0, dilation_rate=1)

        self.conv_6x6 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=3, pad=6, dilation_rate=6)

        self.conv_12x12 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=3, pad=12, dilation_rate=12)

        self.conv_18x18 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=3, pad=18, dilation_rate=18)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channles, out_channels=out_channles,
                kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.final_conv = Atrous_Convolution(
            input_channels=out_channles * 5, output_channels=out_channles,
            kernel_size=1, pad=0, dilation_rate=1)

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool_opt = self.image_pool(x)
        img_pool_opt = F.interpolate(
            img_pool_opt, size=x_18x18.size()[2:],
            mode='bilinear', align_corners=True)
        concat = torch.cat(
            (x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt),
            dim=1)
        x_final_conv = self.final_conv(concat)
        return x_final_conv
    

import torchmetrics
import pytorch_lightning as pl
from metrics import DiceBCELoss

class IOU(torchmetrics.Metric):

    def __init__(self, smooth=1):
        super(IOU, self).__init__()
        self.smooth = smooth
        
        # To accumulate intersection and union values over batches with correct dtype
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        # Sigmoid activation if needed
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        # Update state values
        self.intersection += intersection
        self.union += union

    def compute(self):
        # Final IoU computation
        IoU = (self.intersection + self.smooth) / (self.union + self.smooth)
        return IoU


class Deeplabv3Plus(pl.LightningModule):
    def __init__(self, num_classes , learning_rate):

        super(Deeplabv3Plus, self).__init__()
        
        self.lr = learning_rate

        self.backbone = ResNet_50(output_layer='layer3')

        self.low_level_features = ResNet_50(output_layer='layer1')

        self.assp = ASSP(in_channles=1024, out_channles=256)

        self.conv1x1 = Atrous_Convolution(
            input_channels=256, output_channels=48, kernel_size=1,
            dilation_rate=1, pad=0)

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifer = nn.Conv2d(256, num_classes, 1)
        
        self.loss_fn = DiceBCELoss()
        
        self.metric_iou = IOU()
        
        self.accuracy = torchmetrics.Accuracy(task="binary")

        self.f1_score = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        x_backbone = self.backbone(x)
        x_low_level = self.low_level_features(x)
        x_assp = self.assp(x_backbone)
        
        x_assp_upsampled = F.interpolate(
            x_assp, scale_factor=4.0,  # Change here
            mode='bilinear', align_corners=True)
        
        x_conv1x1 = self.conv1x1(x_low_level)
        x_cat = torch.cat([x_conv1x1, x_assp_upsampled], dim=1)
        x_3x3 = self.conv_3x3(x_cat)


        x_3x3_upscaled = F.interpolate(
            x_3x3, scale_factor=4.0,  # Change here
            mode='bilinear', align_corners=True)

        x_out = self.classifer(x_3x3_upscaled)
        return x_out
    
    def training_step(self,batch,batch_idx):
        data, targets = batch
        targets = targets.float().unsqueeze(1)
        predictions = self.forward(data)
        loss = self.loss_fn(predictions , targets)
        iou = self.metric_iou(predictions , targets)
        
        
        accuracy = self.accuracy(predictions, targets)
        f1_score = self.f1_score(predictions, targets)
        
        self.log_dict(
        {
        "train_loss": loss,
        "iou_(torchmetrics)": iou,
        "acc_(torchmetrics)" : accuracy,
        "f1_score_(torchmetrics)" : f1_score
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        sync_dist=True
        )

        return {"loss":loss ,"iou":iou}
    
    
    def validation_step(self, batch,batch_idx) -> STEP_OUTPUT | None:
        
        data, targets = batch
        targets = targets.float().unsqueeze(1)
        predictions = self.forward(data)
        loss = self.loss_fn(predictions , targets)
        iou = self.metric_iou(predictions , targets)
        
        
        accuracy = self.accuracy(predictions, targets)
        f1_score = self.f1_score(predictions, targets)
        
        self.log_dict(
        {
        "val_loss": loss,
        "iou_(torchmetrics)": iou,
        "acc_(torchmetrics)" : accuracy,
        "f1_score_(torchmetrics)" : f1_score
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        sync_dist=True
        )

        return {"loss":loss ,"iou":iou}
 
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)