import torch.nn as nn
import segmentation_models_pytorch as smp


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

    # Erase Sigmoid
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])

    # Train model in multiple GPUs
    # model = nn.DataParallel(model)
    
    return model.to(DEVICE)