import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pprint

from tqdm import tqdm
from utility.log import log_results, log_results_with_angle, log_terminal
from utility.train import set_parameters, rmse, geom_element, angle_element
from utility.visualization import visualize

def train_function(args, DEVICE, model, loss_fn_pixel, optimizer, train_loader):
    total_loss = 0
    model.train()
    for image, label, _, _, label_list in tqdm(train_loader):
        image = image.to(device=DEVICE)
        label = label.float().to(device=DEVICE)
        
        prediction = model(image)
        loss = loss_fn_pixel(prediction, label) * args.pixel_loss_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def validate_function(args, DEVICE, model, epoch, val_loader, visualize_bool):
    print("=====Starting Validation=====")
    model.eval()

    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    rmse_list = [[0]*len(val_loader) for _ in range(args.output_channel)]
    angle_list = [[0]*len(val_loader) for _ in range(len(args.label_for_angle))]

    with torch.no_grad():
        for idx, (image, label, image_path, image_name, label_list) in enumerate(tqdm(val_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            
            prediction = model(image)
            
            # validate angle difference
            if args.label_for_angle != []:
                predict_angle, label_angle = angle_element(args, prediction, label_list, DEVICE)
                for i in range(len(args.label_for_angle)):
                    angle_list[i][idx] = abs(label_angle[i] - predict_angle[i])

            # validate mean geom difference
            predict_spatial_mean, label_spatial_mean = geom_element(torch.sigmoid(prediction), label)

            ## get rmse difference
            rmse_list, index_list = rmse(args, prediction, label_list, idx, rmse_list)
            extracted_pixels_list.append(index_list)

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()
            dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum() + 1e-8)

            ## visualize
            if epoch % args.dilation_epoch == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1) or epoch % 50 == 0 or visualize_bool:
                if not args.no_visualization:
                    visualize(
                        args, idx, image_path, image_name, label, label_list, epoch, extracted_pixels_list, prediction, prediction_binary,
                        predict_spatial_mean, label_spatial_mean, None, 'train'
                    )
    dice = dice_score/len(val_loader)

    # Removing RMSE for annotation that does not exist in the label
    rmse_mean_by_label = []
    for i in range(len(rmse_list)):
        tmp_sum, count = 0, 0
        for j in range(len(rmse_list[i])):
            if rmse_list[i][j] != -1:
               tmp_sum += rmse_list[i][j]
               count += 1
        rmse_mean_by_label.append(tmp_sum/count)

    total_rmse_mean = sum(rmse_mean_by_label)/len(rmse_mean_by_label)
    print(f"Dice score: {dice}")
    print(f"Average Pixel to Pixel Distance: {total_rmse_mean}")

    if args.label_for_angle != []:
        # add up angle values
        angle_value = []
        for i in range(len(args.label_for_angle)):
            angle_value.append(sum(angle_list[i]))
        angle_value.append(sum(list(map(sum, angle_list))))

        return dice, total_rmse_mean, rmse_list, rmse_mean_by_label, angle_value
    else:
        return dice, total_rmse_mean, rmse_list, rmse_mean_by_label, 0


def train_until(args, model, DEVICE):
    best_loss, best_rmse_mean, best_angle_diff = np.inf, np.inf, np.inf
    best_model = None
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    epoch = 0
    terminate_epoch = 0
    while terminate_epoch != 350:
        visualize_bool = False
        print(f"\nRunning Epoch # {epoch}")
        
        if epoch % args.dilation_epoch == 0:
            args, loss_fn, train_loader, val_loader = set_parameters(args, model, epoch, DEVICE)
        else:
            if dice > args.train_threshold:
                visualize_bool = True
                args.train_threshold -= 0.05
                args, loss_fn, train_loader, val_loader = set_parameters(args, model, epoch, DEVICE)
                terminate_epoch = 0

        loss = train_function(
            args, DEVICE, model, loss_fn, optimizer, train_loader
        )
        dice, rmse_mean, rmse_list, rmse_mean_by_label, angle_value = validate_function(
            args, DEVICE, model, epoch, val_loader, visualize_bool
        )

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
            best_rmse_list = rmse_list
            best_model = model.state_dict()
        
        if args.label_for_angle != []:
            if best_angle_diff > angle_value[len(args.label_for_angle)]:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                }
                # torch.save(checkpoint, f'./results/{args.wandb_name}/best_angle.pth')
                best_angle_diff = angle_value[len(args.label_for_angle)]

        if args.wandb and args.label_for_angle != []:
            log_results_with_angle(
                loss, dice, rmse_mean, best_rmse_mean, rmse_mean_by_label, best_angle_diff, angle_value,
                len(train_loader), len(val_loader), len(args.label_for_angle)
            )
        elif args.wandb and args.label_for_angle == []:
            log_results(
                loss, dice, rmse_mean, best_rmse_mean, rmse_mean_by_label,
                len(train_loader), len(val_loader)
            )
        epoch += 1
        terminate_epoch += 1

    log_terminal(args, 'best_rmse', best_rmse_list)

    return best_model
