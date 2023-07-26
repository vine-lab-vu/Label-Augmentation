import os
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import mean_squared_error as mse
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
    # if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/heatmap'):
    #     os.mkdir(f'{args.result_directory}/{args.wandb_name}/heatmap')
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
