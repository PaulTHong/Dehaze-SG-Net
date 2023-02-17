import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2

from common import config

random.seed(12345)
IMG_TYPES = ['*.jpg', '*.png']

def populate_train_list(orig_images_path, hazy_images_path):
    train_list = []
    val_list = []
    image_list_haze = []
    for img_type in IMG_TYPES:
        image_list_haze.extend(glob.glob(hazy_images_path + img_type))
    tmp_dict = {}

    for image in image_list_haze:
        image = image.split("/")[-1]
        key = image.split("_")[0] + os.path.splitext(image)[-1]
        if key in tmp_dict.keys():
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

    train_keys = []
    val_keys = []

    len_keys = len(tmp_dict.keys())
    for i in range(len_keys):
        if i < len_keys*9/10:  # config.train_num
            train_keys.append(list(tmp_dict.keys())[i])
        else:
            val_keys.append(list(tmp_dict.keys())[i])

    for key in list(tmp_dict.keys()):
        if key in train_keys:
            for hazy_image in tmp_dict[key]:
                train_list.append([orig_images_path + key, hazy_images_path + hazy_image])
        else:
            for hazy_image in tmp_dict[key]:
                val_list.append([orig_images_path + key, hazy_images_path + hazy_image])

    random.shuffle(train_list)
    random.shuffle(val_list)

    return train_list, val_list


class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train'):

        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = cv2.imread(data_orig_path)
        data_hazy = cv2.imread(data_hazy_path)
        data_orig = cv2.cvtColor(data_orig, cv2.COLOR_BGR2RGB)
        data_hazy = cv2.cvtColor(data_hazy, cv2.COLOR_BGR2RGB)

        data_orig = cv2.resize(data_orig, (config.height, config.width), cv2.INTER_LINEAR) / 255.0
        data_hazy = cv2.resize(data_hazy, (config.height, config.width), cv2.INTER_LINEAR) / 255.0
    
        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

