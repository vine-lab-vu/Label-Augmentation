"""
reference:
    Height and Width of image, mask or masks should be equal -> is_check_shapes: 
        https://discuss.pytorch.org/t/height-and-width-of-image-mask-or-masks-should-be-equal-you-can-disable-shapes-check-by-setting-a-parameter-is-check-shapes-false-of-compose-class-do-it-only-if-you-are-sure-about-your-data-consistency/172900
"""

import torch
import pandas as pd
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utility.dataset import dilate_pixel


class CustomDataset(Dataset):
    def __init__(self, args, df, data_type, transform=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index()
        self.data_type = data_type
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.df = self.df.fillna(0)
        image_name = self.df['image_name'][idx]
        image_path = f'{self.args.image_padded_path}/{self.data_type}/{image_name}'
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.data_type == 'train':
            label_list, masks = [0] * (self.args.output_channel*2), []
            for i in range(self.args.output_channel):
                y = int(round(self.df[f'label_{i}_y'][idx]))
                x = int(round(self.df[f'label_{i}_x'][idx]))
                tmp_mask = np.zeros([self.args.image_resize, self.args.image_resize])

                if y != 0 and x != 0:
                    label_list[2*i] = y
                    label_list[2*i+1] = x 
                    tmp_mask = dilate_pixel(self.args, tmp_mask, label_list[2*i], label_list[2*i+1])
                
                masks.append(tmp_mask)

            if self.transform:
                augmentations = self.transform(image=image, masks=masks)
                image = augmentations["image"]
                for i in range(self.args.output_channel):
                    masks[i] = augmentations["masks"][i]

            return image, torch.stack(masks, dim=0), image_path, image_name, label_list
        
        else:
            if self.transform:
                augmentations = self.transform(image=image)
                image = augmentations["image"]

            return image, image_path, image_name


def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size
    DATASET_RATIO = args.dataset_split

    train_val_df = pd.read_csv(args.train_csv_preprocessed)
    split_point1 = int((len(train_val_df)*DATASET_RATIO)/10)
    split_point2 = int((len(train_val_df)*(DATASET_RATIO+2))/10)

    train_df_1 = train_val_df[:split_point1]
    train_df_2 = train_val_df[split_point2:]

    train_df = pd.concat([train_df_1, train_df_2])
    val_df   = train_val_df[split_point1:split_point2]
    test_df  = pd.read_csv(args.test_csv_preprocessed)

    transform = A.Compose([
        A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ], is_check_shapes=False)

    train_dataset = CustomDataset(
        args, train_df, 'train', transform
    )
    val_dataset = CustomDataset(
        args, val_df, 'train', transform
    )
    test_dataset = CustomDataset(
        args, test_df, 'test', transform
    )
    print('len of train dataset: ', len(train_dataset))
    print('len of val dataset: ', len(val_dataset))
    print('len of test dataset: ', len(test_dataset))

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )

    print("---------- Loading Dataset Done ----------")

    return train_loader, val_loader, test_loader