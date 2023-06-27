# import argparse

# from dataset import load_data
# from utility.main import arg_as_list


# def main(args):
#     train, val, test = load_data(args)

#     for a,b,c,d in train:
#         print(a,b,c,d)
#         break


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     ## boolean arguments
#     parser.add_argument('--preprocess', action='store_true')
#     parser.add_argument('--no_image_save', action='store_true', help='whether to save image or not')

#     ## get dataset
#     parser.add_argument('--excel_path', type=str, default="./xlsx/dataset.xlsx", help='path to dataset excel file')

#     ## data preprocessing
#     parser.add_argument('--dicom_path', type=str, default="./data/dicom", help='path to dicom dataset')
#     parser.add_argument('--image_path', type=str, default="./data/image", help='path to image dataset')
#     parser.add_argument('--image_padded_path', type=str, default="./data/image_padded", help='path to padded image')

#     ## hyperparameters - data
#     parser.add_argument('--dataset_csv', type=str, default="./data/xlsx/dataset.csv", help='train csv file path')
#     parser.add_argument('--train_csv', type=str, default="./data/xlsx/train_dataset.csv", help='train csv file path')
#     parser.add_argument('--test_csv', type=str, default="./data/xlsx/test_dataset.csv", help='test csv file path')
#     parser.add_argument('--train_csv_preprocessed', type=str, default="./data/xlsx/train_dataset_preprocessed.csv", help='train csv file path')
#     parser.add_argument('--test_csv_preprocessed', type=str, default="./data/xlsx/test_dataset_preprocessed.csv", help='test csv file path')
#     parser.add_argument('--train_label_txt', type=str, default="./data/txt/train_label.txt", help='train label text file path')
#     parser.add_argument('--test_label_txt', type=str, default="./data/txt/test_label.txt", help='test label text file path')
    
#     parser.add_argument('--dataset_split', type=int, default=8, help='dataset split ratio')
#     parser.add_argument('--dilate', type=int, default=65, help='dilate iteration')
#     parser.add_argument('--dilation_decrease', type=int, default=5, help='dilation decrease in progressive erosion')
#     parser.add_argument('--dilation_epoch', type=int, default=50, help='dilation per epoch')
#     parser.add_argument('--image_resize', type=int, default=512, help='image resize value')
#     parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
#     ## hyperparameters - model
#     parser.add_argument('--seed', type=int, default=2023, help='seed customization for result reproduction')
#     parser.add_argument('--input_channel', type=int, default=3, help='input channel size for UNet')
#     parser.add_argument('--output_channel', type=int, default=20, help='output channel size for UNet')
#     parser.add_argument('--encoder_depth', type=int, default=5, help='model depth for UNet')
#     parser.add_argument("--decoder_channel", type=arg_as_list, default=[256,128,64,32,16], help='model decoder channels')
#     parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
#     parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
#     parser.add_argument('--angle_loss_weight', type=int, default=1, help='weight of the loss function')

#     ## hyperparameters - results
#     parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary prediction')

#     ## wandb
#     parser.add_argument('--wandb', action='store_true', help='whether to use wandb or not')
#     parser.add_argument('--wandb_project', type=str, default="hip replacement", help='wandb project name')
#     parser.add_argument('--wandb_entity', type=str, default="yehyun-suh", help='wandb entity name')
#     parser.add_argument('--wandb_name', type=str, default="temporary", help='wandb name')

#     args = parser.parse_args()
#     main(args)




# import numpy as np 
# mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7 = np.zeros([512,512]),np.zeros([512,512]),np.zeros([512,512]), np.zeros([512,512]),np.zeros([512,512]),np.zeros([512,512]),np.zeros([512,512]),np.zeros([512,512])
# mask = [mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7]
# print(np.array(mask).shape)

import pandas as pd
import numpy as np

df = pd.concat([pd.read_csv('./data/xlsx/train_dataset.csv'), pd.read_csv('./data/xlsx/test_dataset.csv')])
print(df)



