import torch
import os

from tqdm import tqdm

from dataset import load_data
from model import UNet
from utility.visualization import visualize
from utility.train import extract_pixel


def test(args, DEVICE):
    print("=====Starting Testing Process=====")
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/test'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/test')
    _, _, test_loader = load_data(args)

    path = f'./results/{args.wandb_name}/best.pth'
    checkpoint = torch.load(path)
    model = UNet(args, DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    extracted_pixels_list = []
    with torch.no_grad():
        for idx, (image, image_path, image_name) in enumerate(tqdm(test_loader)):
            image = image.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            prediction = model(image)

            ## extract the pixel with highest probability value
            index_list = extract_pixel(args, prediction)
            extracted_pixels_list.append(index_list)

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()

            ## visualize
            visualize(
                args, idx, image_path, image_name, None, None, 
                extracted_pixels_list, prediction, prediction_binary,
                None, None, 'test'
            )
    print("=====Testing Process Done=====")