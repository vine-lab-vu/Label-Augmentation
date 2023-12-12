import os
import torch
import torch.nn as nn
import numpy as np
import math

from sklearn.metrics import mean_squared_error as mse
from scipy import ndimage
from dataset import load_data


class SpatialMean_CHAN(nn.Module):
    """ 
    Spatial Mean CHAN (mean coordinate of non-neg. weight tensor)
    This computes offset from CENTER of 0th index of each channel.

    INPUT: Tensor [ Batch x Channels x H x W x D ]
    OUTPUT: Tensor [ BATCH x Channels x 3]
    """

    def __init__(self, input_shape, eps=1e-8, pytorch_order=True, return_power=False, **kwargs):
        super(SpatialMean_CHAN, self).__init__(**kwargs)

        self.eps = eps
        self.size_in = input_shape
        self.coord_idx_list = []
        self.input_shape_nb_nc = input_shape[1:] # [512,512]
        self.n_chan = input_shape[0] # num of labels

        for idx, in_shape in enumerate(self.input_shape_nb_nc): # [512,512]
            coord_idx_tensor = torch.arange(0, in_shape)        # torch.arange(0,512) -> [0,1,...,511], torch.Size([512])
            coord_idx_tensor = torch.reshape(
                coord_idx_tensor,
                [in_shape] + [1]*(len(self.input_shape_nb_nc)-1) 
            ) # torch.Size([512, 1])

            # [[0,0,..,0],[1,1,..,1],..,[511,511,..,511]]
            coord_idx_tensor = coord_idx_tensor.repeat(
                *([1] + self.input_shape_nb_nc[:idx] + self.input_shape_nb_nc[idx+1:])
            ) # torch.Size([512, 512])

            coord_idx_tensor = coord_idx_tensor.permute(
                *(list(range(1, idx+1)) + [0] + list(range(idx+1, len(self.input_shape_nb_nc))))
            ) # torch.Size([512, 512])

            self.coord_idx_list.append(
                torch.reshape(coord_idx_tensor, [-1])
            ) # torch.Size([262144])

        # coord_idx_list[0] = [0,0,..,0,1,1,..,1,..,511,511,..,511]
        # coord_idx_list[1] = [0,1,..,511,0,1,..,511,..,0,1,..,511]
        self.pytorch_order = pytorch_order
        if pytorch_order:
            self.coord_idx_list.reverse()
        # coord_idx_list[0] = [0,1,..,511,0,1,..,511,..,0,1,..,511]
        # coord_idx_list[1] = [0,0,..,0,1,1,..,1,..,511,511,..,511]

        self.coord_idxs = torch.stack(self.coord_idx_list)    # torch.Size([2, 262144]), list -> Tensor
        self.coord_idxs = torch.unsqueeze(self.coord_idxs, 0) # torch.Size([1, 2, 262144])
        self.coord_idxs = torch.unsqueeze(self.coord_idxs, 0) # torch.Size([1, 1, 2, 262144])
        # coord_idxs[0][0][0] = [0,1,..,511,0,1,..,511,..,0,1,..,511]
        # coord_idxs[0][0][1] = [0,0,..,0,1,1,..,1,..,511,511,..,511]   
        self.return_power = return_power

    def _apply_coords(self, x, verbose=False):
        if verbose:
            print(x.shape)
            print(self.coord_idxs.shape)

        x = torch.unsqueeze(x, 2)
        if verbose:
            print(x.shape)
        
        self.coord_idxs = self.coord_idxs.to(device='cuda')
        numerator = torch.sum(x*self.coord_idxs, dim=[3])        
        denominator = torch.sum(torch.abs(x.detach()) + self.eps, dim=[3])

        if verbose:
            print(numerator.shape)
            print(denominator.shape)

        return numerator / denominator

    def forward(self, x):
        x = torch.reshape(x, [-1, self.n_chan, np.prod(self.input_shape_nb_nc)])
        x = torch.abs(x)
        outputs = self._apply_coords(x)
      
        if self.return_power:
            power_by_chan = x.sum(dim=2, keepdim=True)
            return outputs, power_by_chan
        return outputs 

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.coord_idxs = self.coord_idxs.to(*args, **kwargs)
        for idx in range(len(self.coord_idx_list)):
            self.coord_idx_list[idx].to(*args, **kwargs)
        return self


def create_directories(args):
    if not os.path.exists(f'{args.result_directory}'):
        os.mkdir(f'{args.result_directory}')
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}')
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/label'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/label')
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/pred_w_gt'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/pred_w_gt')
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/heatmap'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/heatmap')
    # for i in range(args.output_channel):
    #     if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/heatmap/label{i}'):
    #         os.mkdir(f'{args.result_directory}/{args.wandb_name}/heatmap/label{i}') 


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
    # num_of_dil_pixels = calculate_number_of_dilated_pixel(args.dilate)
    mask = np.zeros((args.image_resize, args.image_resize))
    mask[int(args.image_resize/2)][int(args.image_resize/2)] = 1.0
    struct = ndimage.generate_binary_structure(rank=2, connectivity=args.connectivity)
    dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=args.dilate).astype(mask.dtype)
    num_of_dil_pixels = np.sum(dilated_mask == 1)

    if args.no_reweight:
        weight = 1
    else:
        weight = ((image_size * 100)/(num_of_dil_pixels))/((image_size * 100)/(image_size - num_of_dil_pixels))
    print(f"Current weight for positive values is {weight}")
    if args.train_until:
        print(f"Current model dice score threshold is {args.train_threshold}")

    loss_fn_pixel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=DEVICE))
    
    return args, loss_fn_pixel, train_loader, val_loader


def extract_pixel(args, prediction): 
    index_list = []
    for i in range(len(prediction)):
        tmp_list = []
        for j in range(args.output_channel):
            index = (prediction[0][j] == torch.max(prediction[0][j])).nonzero()
            if len(index) > 1:
                index = torch.Tensor([[sum(index)[0]/len(index), sum(index)[1]/len(index)]])
            tmp_list.append([index[0].detach().cpu()[0].item(), index[0].detach().cpu()[1].item()])
        index_list.append(tmp_list)

    ## (batch_size, output_channel, 2)
    return index_list


def rmse(args, prediction, label_list, idx, rmse_list):
    index_list = extract_pixel(args, prediction)
    index_list = torch.Tensor(np.array(index_list)).squeeze(0).reshape(args.output_channel*2,1)
    label_list_reshape = np.array(torch.Tensor(label_list), dtype=object).reshape(args.output_channel*2,1)
    label_list_reshape = np.ndarray.tolist(label_list_reshape)

    ## squared=False for RMSE values
    # rmse_value = mse(index_list, label_list_reshape, squared=False) 
    for i in range(args.output_channel):
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])

        if y != 0 and x != 0:
            rmse_list[i][idx] = mse(index_list[2*i:2*(i+1)], label_list_reshape[2*i:2*(i+1)], squared=False)
        else:
            rmse_list[i][idx] = -1

    return rmse_list, extract_pixel(args, prediction)


def calculate_angle(coordinates):
    y1, x1 = coordinates[0], coordinates[1]
    y2, x2 = coordinates[2], coordinates[3]
    y3, x3 = coordinates[4], coordinates[5]

    numerator = y3 - y2
    denominator = x3 - x2
    if denominator == 0:
        denominator = 1e-8
    theta1 = math.degrees(math.atan(numerator/denominator))

    numerator = y1 - y2
    denominator = x1 - x2
    if denominator == 0:
        denominator = 1e-8
    theta2 = math.degrees(math.atan(numerator/denominator))
    theta = abs(theta2 - theta1)

    if theta > 90:
        return 180 - theta
    else:
        return theta


def geom_element(prediction_sigmoid, label):
    predict_spatial_mean_function = SpatialMean_CHAN(list(prediction_sigmoid.shape[1:]))
    predict_spatial_mean          = predict_spatial_mean_function(prediction_sigmoid)
    label_spatial_mean_function   = SpatialMean_CHAN(list(label.shape[1:]))
    label_spatial_mean            = label_spatial_mean_function(label)

    for i in range(label_spatial_mean.shape[0]):
        for j in range(label_spatial_mean.shape[1]):
            if int(label_spatial_mean[i][j][0]) == 0 and int(label_spatial_mean[i][j][1]) == 0:
                predict_spatial_mean[i][j][0] = 0
                predict_spatial_mean[i][j][1] = 0

    return predict_spatial_mean, label_spatial_mean


def angle_element(args, prediction, label_list, DEVICE):
    index_list = extract_pixel(args, prediction)
    label_sorted_list = []
    for i in range(len(label_list[0])):
        tmp_list = []
        for j in range(0,len(label_list),2):
            tmp_list.append([label_list[j][i].item(), label_list[j+1][i].item()])
        label_sorted_list.append(tmp_list)

    angle_preds, angle_label = [], []
    for i in range(len(index_list)):
        for j in range(len(args.label_for_angle)):
            coord_preds, coord_label = [], []
            for k in range(len(args.label_for_angle[j])):
                coord_preds.append(index_list[i][args.label_for_angle[j][k]][0])
                coord_preds.append(index_list[i][args.label_for_angle[j][k]][1])
                coord_label.append(label_sorted_list[i][args.label_for_angle[j][k]][0])
                coord_label.append(label_sorted_list[i][args.label_for_angle[j][k]][1])
            angle_preds.append(calculate_angle(coord_preds))
            angle_label.append(calculate_angle(coord_label))

    return angle_preds, angle_label
