import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pprint

from tqdm import tqdm
from utility.log import log_results, log_terminal
from utility.train import set_parameters, SpatialMean_CHAN, extract_pixel, rmse
from utility.visualization import visualize

def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader):
    total_loss, total_pixel_loss, total_geom_loss = 0, 0, 0
    total_num_noma, total_num_pred_noma = 0, 0
    model.train()

    for image, label, _, _, label_list in tqdm(train_loader):
        image = image.to(device=DEVICE)
        label = label.float().to(device=DEVICE)
        
        prediction = model(image)
        prediction_sigmoid = torch.sigmoid(prediction)

        ## Pixel Loss
        loss_pixel = loss_fn_pixel(prediction, label)

        ## Geometry Loss
        predict_spatial_mean_function = SpatialMean_CHAN(list(prediction.shape[1:]))
        predict_spatial_mean          = predict_spatial_mean_function(prediction_sigmoid)
        label_spatial_mean_function   = SpatialMean_CHAN(list(label.shape[1:]))
        label_spatial_mean            = label_spatial_mean_function(label)

        for i in range(label_spatial_mean.shape[0]):
            for j in range(label_spatial_mean.shape[1]):
                if int(label_spatial_mean[i][j][0]) == 0 and int(label_spatial_mean[i][j][1]) == 0:
                    predict_spatial_mean[i][j][0] = 0
                    predict_spatial_mean[i][j][1] = 0

        loss_geometry = loss_fn_geometry(predict_spatial_mean, label_spatial_mean)

        if args.geom_loss: loss = loss_pixel + args.geom_loss_weight * loss_geometry
        else:              loss = loss_pixel

        ## NoMa Loss
        num_noma, num_pred_noma = 0, 0
        for i in range(len(label_list[0])): # 0~17
            for j in range(0,len(label_list),2):# 0~39
                if label_list[j][i].item() == 0 and label_list[j+1][i].item() == 0:
                    num_noma += 1
                    # print(i,j//2, end='\t')
                    # print(torch.max(prediction_sigmoid[i][j//2]).item()) # 18, 20, 512, 512
                    if torch.max(prediction_sigmoid[i][j//2]).item() > args.threshold:
                        num_pred_noma += 1

        total_num_noma += num_noma
        total_num_pred_noma += num_pred_noma
        
        if num_noma == 0: loss_noma = 0
        else:             loss_noma = num_pred_noma/num_noma

        if args.noma_loss:
            loss = loss * (1 + loss_noma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss          += loss.item()
        total_pixel_loss    += loss_pixel.item() 
        total_geom_loss     += loss_geometry.item()
        total_num_noma      += num_noma
        total_num_pred_noma += num_pred_noma

    return total_loss, total_pixel_loss, total_geom_loss, total_num_pred_noma/total_num_noma


def validate_function(args, DEVICE, model, epoch, val_loader):
    print("=====Starting Validation=====")
    model.eval()

    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    rmse_list = [[0]*len(val_loader) for _ in range(args.output_channel)]

    with torch.no_grad():
        label_list_total, nomas_total = [], []
        for idx, (image, label, image_path, image_name, label_list) in enumerate(tqdm(val_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            label_list_total.append(label.detach().cpu().numpy())
            
            prediction = model(image)

            predict_spatial_mean_function = SpatialMean_CHAN(list(prediction.shape[1:]))
            predict_spatial_mean          = predict_spatial_mean_function(torch.sigmoid(prediction))
            label_spatial_mean_function   = SpatialMean_CHAN(list(label.shape[1:]))
            label_spatial_mean            = label_spatial_mean_function(label)

            ## extract the pixel with highest probability value
            index_list = extract_pixel(args, prediction)
            rmse_list = rmse(
                args, index_list, label_list, idx, rmse_list
            )
            extracted_pixels_list.append(index_list)

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()
            dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum() + 1e-8)

            ## visualize
            if epoch % args.dilation_epoch == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1):
                if not args.no_visualization:
                    visualize(
                        args, idx, image_path, image_name, label_list, epoch, extracted_pixels_list, prediction, prediction_binary,
                        predict_spatial_mean, label_spatial_mean, 'train'
                    )
            # elif epoch % (args.epochs // 10) == 0:
            #     visualize(
            #         args, idx, image_path, image_name, label_list, epoch, extracted_pixels_list, prediction, prediction_binary,
            #         predict_spatial_mean, label_spatial_mean, 'train'
            #     )

    dice = dice_score/len(val_loader)

    rmse_sum = 0
    for i in range(len(rmse_list)):
        for j in range(len(rmse_list[i])):
            if rmse_list[i][j] != -1:
                rmse_sum += rmse_list[i][j]

    rmse_mean = rmse_sum/(len(val_loader)*args.output_channel)
    print(f"Dice score: {dice}")
    print(f"Average Pixel to Pixel Distance: {rmse_mean}")

    return dice, rmse_mean, rmse_list


def train(args, model, DEVICE):
    best_loss, best_rmse_mean = np.inf, np.inf
    loss_fn_geometry = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")
        
        if epoch % args.dilation_epoch == 0:
            args, loss_fn_pixel, train_loader, val_loader = set_parameters(
                args, model, epoch, DEVICE
            )

        loss, loss_pixel, loss_geom, loss_noma = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader
        )
        dice, rmse_mean, rmse_list = validate_function(
            args, DEVICE, model, epoch, val_loader
        )

        print("No Marker Prediction Percentage: ", loss_noma)
        print("Average Train Loss: ", loss/len(train_loader))
        if best_loss > loss:
            print("=====New best model=====")
            best_loss = loss

        if best_rmse_mean > rmse_mean:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }
            # torch.save(checkpoint, f'./results/{args.wandb_name}/best.pth')
            best_rmse_mean = rmse_mean

        if args.wandb:              
            log_results(
                loss, loss_pixel, loss_geom, loss_noma,
                dice, rmse_mean, best_rmse_mean, rmse_list, 
                len(train_loader), len(val_loader)
            )
    log_terminal(args, rmse_list)