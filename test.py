import torch 
import math
import time
import os 
import csv
import numpy as np

from dataset import load_data
from tqdm import tqdm
from utility.log import log_terminal
from utility.train import rmse, geom_element, angle_element
from utility.visualization import visualize


def test(args, model, DEVICE):
    print("=====Starting Testing Process=====")
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/test'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/test')
    _, _, test_loader = load_data(args)

    model.eval()
    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    extracted_pixels_to_df = []
    rmse_list = [[0]*len(test_loader) for _ in range(args.output_channel)]
    angle_list = [[0]*len(test_loader) for _ in range(len(args.label_for_angle))]
    angle_total = []
    label_total = []

    extracted_pixels_list = []
    with torch.no_grad():
        start = time.time()
        for idx, (image, label, image_path, image_name, label_list) in enumerate(tqdm(test_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            prediction = model(image)
            label_total.append(np.ndarray.tolist(np.array(torch.Tensor(label_list), dtype=object).reshape(args.output_channel*2,1)))

            # validate mean geom difference
            predict_spatial_mean, label_spatial_mean = geom_element(torch.sigmoid(prediction), label)

            ## get rmse difference
            rmse_list, index_list = rmse(args, prediction, label_list, idx, rmse_list)
            extracted_pixels_list.append(index_list)

            extracted_pixels_array = np.array(index_list[0]).reshape(-1)
            tmp_list = [f'{image_name}.png', len(index_list[0])]
            for i in range(len(extracted_pixels_array)):
                tmp_list.append(extracted_pixels_array[i])
            extracted_pixels_to_df.append(tmp_list)

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()

            ## visualize
            visualize(
                args, idx, image_path, image_name, label_list, None, extracted_pixels_list, prediction, prediction_binary,
                predict_spatial_mean, label_spatial_mean, None, 'test'
            )
        end = time.time()

    print("=====Testing Process Done=====")
    print(f"{end - start:.5f} seconds for {len(test_loader)} images")

    log_terminal(args, "test_prediction", extracted_pixels_list)
    log_terminal(args, "test_label", label_total)

    row_name = ["image_name", "number_of_labels"]
    for i in range(args.output_channel):
        row_name.append(f'label_{i}_y')
        row_name.append(f'label_{i}_x')
    csv_path = f'results/{args.wandb_name}/{args.wandb_name}_test_prediction.csv'
    with open(csv_path, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(row_name)
        write.writerows(sorted(extracted_pixels_to_df))
