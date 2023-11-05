import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from PIL import Image


def visualize(
        args, idx, image_path, image_name, label_list, epoch, 
        extracted_pixels_list, prediction, prediction_binary,
        predict_spatial_mean, label_spatial_mean, angles, mode
    ):
    original_image= Image.open(image_path).resize((args.image_resize,args.image_resize)).convert("RGB")

    if mode == 'train':
        # if epoch == 0:
        #     image_w_label(args, image_name, original_image, label_list)
        # if idx == 0:
        #     image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list, mode)
        image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list, mode)
        image_w_seg_pred(args, idx, image_name, original_image, epoch, prediction_binary, predict_spatial_mean, label_spatial_mean)
        # image_w_heatmap(args, idx, image_name, epoch, prediction)
    else:
        image_w_ground_truth_and_prediction(args, idx, image_name, original_image, None, extracted_pixels_list, label_list, mode)
        # image_w_heatmap(args, image_name, original_image, epoch, prediction_binary)


def image_w_label(args, image_name, original_image, label_list):
    for i in range(args.output_channel):
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])
        
        if y != 0 and x != 0:
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))
        # original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))

    original_image.save(f'{args.result_directory}/{args.wandb_name}/label/{image_name}_label.png')


def image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list, mode):
    for i in range(args.output_channel):
        # if mode != 'test':
        #     y = int(label_list[2*i])
        #     x = int(label_list[2*i+1])

        #     if y != 0 and x != 0:
        #         original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))
        #         y = int(extracted_pixels_list[idx][0][i][0])
        #         x = int(extracted_pixels_list[idx][0][i][1])
        #         original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (255, 0, 0),-1))
        
        # else:
        #     y = int(extracted_pixels_list[idx][0][i][0])
        #     x = int(extracted_pixels_list[idx][0][i][1])
        #     original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (255, 0, 0),-1))
        
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])
        if y != 0 and x != 0:
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))
            y = int(extracted_pixels_list[idx][0][i][0])
            x = int(extracted_pixels_list[idx][0][i][1])
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (255, 0, 0),-1))

    if mode == 'test':
        original_image.save(f'{args.result_directory}/{args.wandb_name}/test/{image_name}.png')
    else:
        original_image.save(f'{args.result_directory}/{args.wandb_name}/pred_w_gt/{image_name}_{epoch}.png')


def image_w_seg_pred(args, idx, image_name, original_image, epoch, prediction_binary, predict_spatial_mean, label_spatial_mean):
    for i in range(args.output_channel):
        background = prediction_binary[0][i].unsqueeze(0)
        background = TF.to_pil_image(torch.cat((background, background, background), dim=0))
        overlaid_image = Image.blend(original_image, background , 0.3)
        x = int(label_spatial_mean[0][i][0])
        y = int(label_spatial_mean[0][i][1])
        overlaid_image = Image.fromarray(cv2.circle(np.array(overlaid_image), (x,y), 8, (0, 0, 255),-1))

        x = int(predict_spatial_mean[0][0][0])
        y = int(predict_spatial_mean[0][0][1])
        overlaid_image = Image.fromarray(cv2.circle(np.array(overlaid_image), (x,y), 8, (255, 0, 0),-1))
        overlaid_image.save(f'{args.result_directory}/{args.wandb_name}/heatmap/label{i}/{image_name}_{epoch}_label{i}.png')


def image_w_heatmap(args, idx, image_name, epoch, prediction):
    for i in range(len(prediction[0])):
        plt.imshow(prediction[0][i].detach().cpu().numpy(), interpolation='nearest')
        plt.axis('off')
        plt.savefig(f'./results/{args.wandb_name}/heatmap/label{i}/{image_name}_{epoch}_heatmap.png', bbox_inches='tight', pad_inches=0, dpi=150)