import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torch.nn import functional as F


def UNet(args, DEVICE):
    print("---------- Loading  Model ----------")

    model = smp.Unet(
        encoder_name     = 'resnet101', 
        encoder_weights  = 'imagenet', 
        encoder_depth    = args.encoder_depth,
        classes          = args.output_channel, 
        activation       = 'sigmoid',
        decoder_channels = args.decoder_channel,
    )
    print("---------- Model Loaded ----------")

    ## Erase Sigmoid
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])
    
    return model.to(DEVICE)