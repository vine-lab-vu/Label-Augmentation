import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from tqdm import tqdm

from utility.log import log_results
from utility.train import set_parameters, extract_pixel, rmse
# from visualization import save_predictions_as_images, box_plot, angle_visualization
from utility.visualization import visualize

def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_angle, optimizer, train_loader):
    total_loss, total_pixel_loss, total_angle_loss = 0, 0, 0
    model.train()

    for image, label, _, _, _ in tqdm(train_loader):
        image = image.to(device=DEVICE)
        label = label.to(device=DEVICE)
        prediction =  model(image)
        
        # calculate log loss with pixel value
        loss_pixel = loss_fn_pixel(prediction, label)

        # # calculate the difference between GT angle and predicted angle
        # angle_pred, angle_gt = [], []
        # for i in range(args.output_channel):
        #     index_list = extract_highest_probability_pixel(args, prediction[i].unsqueeze(0))
        #     angle_pred.append([calculate_angle(args, index_list, "preds")])
        #     angle_gt.append([calculate_angle(args, label_list, "label")])
        # loss_angle = loss_fn_angle(torch.Tensor(angle_pred), torch.Tensor(angle_gt))

        loss = loss_pixel
        # loss = loss_pixel + args.angle_loss_weight * loss_angle 

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss       += loss.item()
        total_pixel_loss += loss_pixel.item() 
        # total_angle_loss += loss_angle.item()

    return total_loss, total_pixel_loss
    # return loss.item(), loss_pixel.item(), loss_angle.item()


def validate_function(args, DEVICE, model, epoch, val_loader):
    print("=====Starting Validation=====")
    model.eval()

    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    rmse_list = [[0]*len(val_loader) for _ in range(args.output_channel)]

    with torch.no_grad():
        label_list_total, angles_total = [], []
        for idx, (image, label, image_path, image_name, label_list) in enumerate(tqdm(val_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            label_list_total.append(label.detach().cpu().numpy())
            
            prediction = model(image)

            ## extract the pixel with highest probability value
            index_list = extract_pixel(args, prediction)
            rmse_value, rmse_list = rmse(
                args, index_list, label_list, idx, rmse_list
            )
            
            extracted_pixels_list.append(index_list)
            rmse_total += rmse_value

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()
            dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum() + 1e-8)

            ## visualize
            visualize(args, idx, image_path, image_name, label_list, epoch, extracted_pixels_list)

    dice = dice_score/len(val_loader)
    mean_rmse = rmse_total/len(val_loader)
    print(f"Dice score: {dice}")
    print(f"Average Pixel to Pixel Distance: {mean_rmse}")

    return dice, mean_rmse


def train(args, model, DEVICE):
    best_loss, best_rmse_mean = np.inf, np.inf
    
    loss_fn_angle = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")
        
        if epoch % args.dilation_epoch == 0:
            args, loss_fn_pixel, train_loader, val_loader = set_parameters(
                args, model, epoch, DEVICE
            )

        loss, loss_pixel = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_angle, optimizer, train_loader
        )
        # loss, loss_pixel, loss_angle = train_function(
        #     args, DEVICE, model, loss_fn_pixel, loss_fn_angle, optimizer, train_loader
        # )
        dice, mean_rmse = validate_function(
            args, DEVICE, model, epoch, val_loader
        )

        print("Average Train Loss: ", loss/len(train_loader))
        if best_loss > loss:
            print("=====New best model=====")
            best_loss = loss
        
        if args.wandb:              
            log_results(loss/len(train_loader), dice, mean_rmse)