import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from PIL import Image


def visualize(args, idx, image_path, image_name, label_list, epoch, extracted_pixels_list, prediction_binary):
    original_image= Image.open(image_path).resize((args.image_resize,args.image_resize)).convert("RGB")

    if epoch == 0:
        image_w_label(args, image_name, original_image, label_list)
    if idx == 0:
        image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list)
        image_w_heatmap(args, image_name, original_image, epoch, prediction_binary)


def image_w_label(args, image_name, original_image, label_list):
    for i in range(args.output_channel):
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])
        
        if y != 0 and x != 0:
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))

    original_image.save(f'{args.result_directory}/{args.wandb_name}/label/{image_name}_label.png')


def image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list):
    for i in range(args.output_channel):
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])

        if y != 0 and x != 0:
            # print(f"Label{i}",image_name, "label_y",y, "label_x",x, end=' ')
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))

            y = int(extracted_pixels_list[idx][i][0][0])
            x = int(extracted_pixels_list[idx][i][0][1])
            # print("label_y",y, "label_x",x)
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (255, 0, 0),-1))
        else:
            # print(f"Label {i} has value (0, 0)")
            pass

    original_image.save(f'{args.result_directory}/{args.wandb_name}/pred_w_gt/{image_name}_{epoch}.png')


def image_w_heatmap(args, image_name, original_image, epoch, prediction_binary):
    for i in range(args.output_channel):
        background = prediction_binary[0][i].unsqueeze(0)
        background = TF.to_pil_image(torch.cat((background, background, background), dim=0))
        overlaid_image = Image.blend(original_image, background , 0.3)
        overlaid_image.save(f'{args.result_directory}/{args.wandb_name}/heatmap/label{i}/{image_name}_{epoch}_label{i}.png')