import os
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import mean_squared_error as mse
from dataset import load_data


def create_directories(args):
    if not os.path.exists(f'{args.result_directory}'):
        os.mkdir(f'{args.result_directory}')
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}')
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/label'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/label')
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/pred_w_gt'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/pred_w_gt')
    
    # if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/annotation'):
    #     os.mkdir(f'{args.result_directory}/{args.wandb_name}/annotation')
    # if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/overlaid'):
    #     os.mkdir(f'{args.result_directory}/{args.wandb_name}/overlaid')
    # for i in range(args.output_channel):
    #     if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/overlaid/label{i}'):
    #         os.mkdir(f'{args.result_directory}/{args.wandb_name}/overlaid/label{i}')
    # for i in range(args.output_channel):
    #     if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/label{i}'):
    #         os.mkdir(f'{args.result_directory}/{args.wandb_name}/label{i}')
    # if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/pth_file'):
    #     os.mkdir(f'{args.result_directory}/{args.wandb_name}/pth_file')
    # if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/angles'):
    #     os.mkdir(f'{args.result_directory}/{args.wandb_name}/angles')


def calculate_number_of_dilated_pixel(k):
    sum = 0
    for i in range(k+1):
        if i == 0: sum += 1
        else:      sum += 4 * i
    return sum


def set_parameters(args, model, epoch, DEVICE):
    create_directories(args)
    train_loader, val_loader, _ = load_data(args)
    if epoch != 0:
        args.dilate = args.dilate - args.dilation_decrease
    if args.dilate < 1:
        args.dilate = 0

    image_size = args.image_resize * args.image_resize
    num_of_dil_pixels = calculate_number_of_dilated_pixel(args.dilate)

    weight = ((image_size * 100)/(num_of_dil_pixels))/((image_size * 100)/(image_size - num_of_dil_pixels))
    print(f"Current weight for positive values is {weight}")

    loss_fn_pixel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=DEVICE))
    
    return args, loss_fn_pixel, train_loader, val_loader


def extract_pixel(args, prediction): 
    index_list = []
    for i in range(args.output_channel):
        index = (prediction[0][i] == torch.max(prediction[0][i])).nonzero()
        if len(index) > 1:
            index = torch.Tensor([[sum(index)[0]/len(index), sum(index)[1]/len(index)]])
        index_list.append(index.detach().cpu().numpy())

    return index_list


def rmse(args, highest_probability_pixels, label_list, idx, rmse_list):
    highest_probability_pixels = torch.Tensor(np.array(highest_probability_pixels)).squeeze(0).reshape(args.output_channel*2,1)
    label_list_reshape = np.array(torch.Tensor(label_list), dtype=object).reshape(args.output_channel*2,1)
    label_list_reshape = np.ndarray.tolist(label_list_reshape)

    ## squared=False for RMSE values
    # rmse_value = mse(highest_probability_pixels, label_list_reshape, squared=False) 
    for i in range(args.output_channel):
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])

        if y != 0 and x != 0:
            rmse_list[i][idx] = mse(highest_probability_pixels[2*i:2*(i+1)], label_list_reshape[2*i:2*(i+1)], squared=False)
        else:
            rmse_list[i][idx] = -1

    return rmse_list
