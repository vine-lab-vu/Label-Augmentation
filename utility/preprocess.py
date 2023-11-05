import cv2
import csv
import numpy as np
import pandas as pd
import os
import shutil

from glob import glob
from PIL import Image
from tqdm import tqdm


def relocate(args):
    print("---------- Relocating Images based on Text Files ----------")

    relocation(args, 'train')
    print("Relocating Training Images Done")
    relocation(args, 'test')
    print("Relocating Test Images Done")

    print("---------- Relocating Images Done ----------")


def create_csv(args):
    print("---------- Starting Creating csv file from txt file ----------")
    if not os.path.exists(f'{args.csv_path}'):
        os.mkdir(f'{args.csv_path}')

    text_to_csv(args, args.train_label_txt, 'train')
    text_to_csv(args, args.test_label_txt, 'test')

    print("---------- Creating csv File Done ----------\n")


def pad_dataset(args):
    print("---------- Starting Padding Process ----------")
    if not os.path.exists(f'{args.image_padded_path}'):
        os.mkdir(f'{args.image_padded_path}')

    train_df = pad(args, pd.read_csv(args.train_csv), 'train')
    test_df = pad(args, pd.read_csv(args.test_csv), 'test')

    dataset_df = pd.concat([train_df, test_df])
    dataset_df.to_csv(args.dataset_csv)

    print("---------- Padding Image Process Done ----------")


def relocation(args, data_type):
    if not os.path.exists(f'{args.image_path}/{data_type}'):
        os.mkdir(f'{args.image_path}/{data_type}')

    with open(f"./data/txt/{data_type}_label.txt", 'r') as f:
        for line in f:
            image_name = line.strip().split(',')[0]
            image_path = f'{args.image_path_all}/{image_name}'
            shutil.copy(image_path, f'{args.image_path}/{data_type}/{image_name}')


def text_to_csv(args, annotation_file, data_type):
    label_list = []
    
    if data_type == 'train':
        csv_path = args.train_csv
        with open(annotation_file, 'r') as f:
            for line in f:
                image_name = line.strip().split(',')[0]
                num_of_labels = int(line.strip().split(',')[1])

                if args.output_channel != num_of_labels:
                    print(f'File {image_name} cannot be converted to csv since it has different number of labels {num_of_labels}')
                else:
                    one_line_list = [image_name, num_of_labels]
                    for i in range(num_of_labels):
                        y = int(line.strip().split(',')[(2*i)+2])
                        x = int(line.strip().split(',')[(2*i)+3])

                        if y < 30 and x < 30:
                            one_line_list.append(0) 
                            one_line_list.append(0)
                        else:
                            one_line_list.append(y)
                            one_line_list.append(x)
                label_list.append(one_line_list)

        row_name = ["image_name","number_of_labels"]
        for i in range(num_of_labels):
            row_name.append(f'label_{i}_y')
            row_name.append(f'label_{i}_x')

    else:
        csv_path = args.test_csv
        with open(annotation_file, 'r') as f:
            for line in f:
                image_name = line.strip().split(',')[0]
                num_of_labels = int(line.strip().split(',')[1])

                if args.output_channel != num_of_labels:
                    print(f'File {image_name} cannot be converted to csv since it has different number of labels {num_of_labels}')
                else:
                    one_line_list = [image_name, num_of_labels]
                    for i in range(num_of_labels):
                        y = int(line.strip().split(',')[(2*i)+2])
                        x = int(line.strip().split(',')[(2*i)+3])

                        if y < 30 and x < 30:
                            one_line_list.append(0) 
                            one_line_list.append(0)
                        else:
                            one_line_list.append(y)
                            one_line_list.append(x)
                label_list.append(one_line_list)

        row_name = ["image_name","number_of_labels"]
        for i in range(num_of_labels):
            row_name.append(f'label_{i}_y')
            row_name.append(f'label_{i}_x')

    # else:
    #     csv_path = args.test_csv
    #     with open(annotation_file, 'r') as f:
    #         for line in f:
    #             image_name = line.strip().split(',')[0]
    #             one_line_list = [image_name]
    #             label_list.append(one_line_list)

        # row_name = ["image_name"]

    with open(csv_path, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(row_name)
        write.writerows(sorted(label_list))


def pad(args, df, data_type):
    if not os.path.exists(f'{args.image_padded_path}/{data_type}'): 
        os.mkdir(f'{args.image_padded_path}/{data_type}')
    image_paths = sorted(glob(f'{args.image_path}/{data_type}/*.png'))

    for image_path in tqdm(image_paths):
        image_name = image_path.split('/')[-1]
        image_array = cv2.imread(image_path)
        fill = abs(image_array.shape[0] - image_array.shape[1])
        row = df.loc[df['image_name'] == image_name].values.tolist()[0]

        ## when height > width
        if image_array.shape[0] > image_array.shape[1]:
            padded_array = np.zeros((
                image_array.shape[0], image_array.shape[0], 3
            ))
        
            for i in range(image_array.shape[0]):
                left = np.zeros((int(fill/2),3))
                middle = np.array(image_array[i])
                if fill % 2 == 0: right = np.zeros((int(fill/2),3))
                else:             right = np.zeros((int(fill/2)+1,3))

                tmp = np.concatenate((left, middle), axis=0)
                padded_array[i] = np.concatenate((tmp, right), axis=0)
            
            ## pad & resize values in csv
            # if data_type == 'train':
            for i in range(row[1]):
                if row[2*i+2] != 0 and row[2*i+3] != 0:
                    row[2*i+3] = row[2*i+3] + left.shape[0]
                    row[2*i+2] = row[2*i+2] * (args.image_resize/padded_array.shape[0])
                    row[2*i+3] = row[2*i+3] * (args.image_resize/padded_array.shape[0])

        ## when height < width
        elif image_array.shape[0] < image_array.shape[1]:
            padded_array = np.zeros((
                image_array.shape[1], image_array.shape[1], 3
            ))

            high = np.zeros((int(fill/2),image_array.shape[1],3))
            if fill % 2 == 0: low = np.zeros((int(fill/2),image_array.shape[1],3))
            else:             low = np.zeros((int(fill/2)+1,image_array.shape[1],3))

            tmp = np.vstack([high, image_array])
            padded_array = np.vstack([tmp, low])

            ## pad & resize values in csv
            # if data_type == 'train':
            for i in range(row[1]):
                if row[2*i+2] != 0 and row[2*i+3] != 0:
                    row[2*i+2] = row[2*i+2] + high.shape[0]
                    row[2*i+2] = row[2*i+2] * (args.image_resize/padded_array.shape[0])
                    row[2*i+3] = row[2*i+3] * (args.image_resize/padded_array.shape[0])
        
        ## when height == width
        else: 
            padded_array = np.zeros((
                image_array.shape[0], image_array.shape[1], 3
            ))
            padded_array = image_array[:]

            ## resize values in csv
            # if data_type == 'train':
            for i in range(row[1]):
                if row[2*i+2] != 0 and row[2*i+3] != 0:
                    row[2*i+2] = row[2*i+2] * (args.image_resize/padded_array.shape[0])
                    row[2*i+3] = row[2*i+3] * (args.image_resize/padded_array.shape[0])
            
        df.loc[df['image_name'] == image_name] = row
        norm = (padded_array - np.min(padded_array)) / (np.max(padded_array) - np.min(padded_array))
        norm_img = np.uint8(norm*255)
        padded_image = Image.fromarray(norm_img)
        padded_image.save(f'{args.image_padded_path}/{data_type}/{image_name}')
        
    if data_type == 'train':
        df.to_csv(args.train_csv_preprocessed)
    else:
        df.to_csv(args.test_csv_preprocessed)

    return df