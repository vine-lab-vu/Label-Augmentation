import cv2
import numpy as np

from PIL import Image


def visualize(args, idx, image_path, image_name, label_list, epoch, extracted_pixels_list):
    original_image= Image.open(image_path).resize((args.image_resize,args.image_resize)).convert("RGB")

    if epoch == 0:
        image_w_label(args, image_name, original_image, image_path, label_list, epoch)
    if epoch % args.dilation_epoch == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1):
        image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list)


def image_w_label(args, image_name, original_image, image_path, label_list, epoch):
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
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))

            y = int(extracted_pixels_list[idx][i][0][0])
            x = int(extracted_pixels_list[idx][i][0][1])
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (255, 0, 0),-1))

    original_image.save(f'{args.result_directory}/{args.wandb_name}/pred_w_gt/{image_name}_{epoch}.png')