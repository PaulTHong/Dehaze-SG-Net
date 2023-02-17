import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF


class NH_HAZE(data.Dataset):
    def __init__(self, dir_path, mode='train', crop=False, crop_h=240, crop_w=320, resize=False, return_seg=False):
        """
        mode = 'train' 'valid' or 'test'
        """
        self.mode = mode
        self.crop = crop
        self.crop_h, self.crop_w = crop_h, crop_w
        self.resize = resize
        self.return_seg = return_seg

        img_paths = os.listdir(dir_path)
        self.hazy_paths = [os.path.join(dir_path, p) for p in img_paths if p[3:7] == 'hazy']  # **_hazy.png
        self.clear_paths = [os.path.join(dir_path, p) for p in img_paths if p[3:5] == 'GT']  # **_GT.png
        self.hazy_paths.sort()
        self.clear_paths.sort()

        if mode == 'train':
            if crop:
                self.hazy_paths = self.hazy_paths[:40]
                self.clear_paths = self.clear_paths[:40]
                # self.hazy_paths = self.hazy_paths[:50]
                # self.clear_paths = self.clear_paths[:50]
            else:
                # Load cropped images directly.
                pass
        else:
            self.hazy_paths = self.hazy_paths[40:45]
            self.clear_paths = self.clear_paths[40:45]
            # self.hazy_paths = self.hazy_paths[50:]
            # self.clear_paths = self.clear_paths[50:]

        print("Number of %s samples: %d" % (mode, len(self.hazy_paths)))

    def __getitem__(self, index):
        hazy_path, clear_path = self.hazy_paths[index], self.clear_paths[index]
        self.hazy_path, self.clear_path = hazy_path, clear_path

        hazy = Image.open(hazy_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')

        if self.mode == 'train':
            if self.crop:
                i, j, h, w = tfs.RandomCrop.get_params(hazy, output_size=(self.crop_h, self.crop_w))
                hazy = FF.crop(hazy, i, j, h, w)
                clear = FF.crop(clear, i, j, h, w)

            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            hazy = tfs.RandomHorizontalFlip(rand_hor)(hazy)
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                hazy = FF.rotate(hazy, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)
        else:
            if self.resize:
                # When test, resize as the size of training images firstly.
                hazy = hazy.resize((self.crop_w, self.crop_h))

        hazy = tfs.ToTensor()(hazy)
        if self.return_seg:
            hazy2seg = tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(hazy)
        hazy = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(hazy)
        clear = tfs.ToTensor()(clear)
        
        if self.return_seg:
            return hazy, clear, hazy2seg
        else:
            return hazy, clear

    def __len__(self):
        return len(self.hazy_paths)

    def get_img_name(self):
        return self.hazy_path, self.clear_path


class DENSE_HAZE(data.Dataset):
    def __init__(self, dir_path, mode='train', crop=False, crop_h=240, crop_w=320, resize=False, return_seg=False):
        """
        mode = 'train' 'valid' or 'test'
        """
        self.mode = mode
        self.crop = crop
        self.crop_h, self.crop_w = crop_h, crop_w
        self.resize = resize
        self.return_seg = return_seg

        hazy_dir = os.path.join(dir_path, 'hazy')
        clear_dir = os.path.join(dir_path, 'GT')
        self.hazy_paths = [os.path.join(hazy_dir, p) for p in os.listdir(hazy_dir)]  # **_hazy.png
        self.clear_paths = [os.path.join(clear_dir, p) for p in os.listdir(clear_dir)]  # **_GT.png
        self.hazy_paths.sort()
        self.clear_paths.sort()

        if mode == 'train':
            if crop:
                self.hazy_paths = self.hazy_paths[:45]
                self.clear_paths = self.clear_paths[:45]
                # self.hazy_paths = self.hazy_paths[:50]
                # self.clear_paths = self.clear_paths[:50]
            else:
                # Load cropped images directly.
                pass
        else:
            self.hazy_paths = self.hazy_paths[45:50]
            self.clear_paths = self.clear_paths[45:50]
            # self.hazy_paths = self.hazy_paths[50:]
            # self.clear_paths = self.clear_paths[50:]

        print("Number of %s samples: %d" % (mode, len(self.hazy_paths)))

    def __getitem__(self, index):
        hazy_path, clear_path = self.hazy_paths[index], self.clear_paths[index]
        self.hazy_path, self.clear_path = hazy_path, clear_path

        hazy = Image.open(hazy_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')

        if self.mode == 'train':
            if self.crop:
                i, j, h, w = tfs.RandomCrop.get_params(hazy, output_size=(self.crop_h, self.crop_w))
                hazy = FF.crop(hazy, i, j, h, w)
                clear = FF.crop(clear, i, j, h, w)

            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            hazy = tfs.RandomHorizontalFlip(rand_hor)(hazy)
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                hazy = FF.rotate(hazy, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)
        else:
            if self.resize:
                # When test, resize as the size of training images firstly.
                hazy = hazy.resize((self.crop_w, self.crop_h))

        hazy = tfs.ToTensor()(hazy)
        if self.return_seg:
            hazy2seg = tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(hazy)
        hazy = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(hazy)
        clear = tfs.ToTensor()(clear)
        
        if self.return_seg:
            return hazy, clear, hazy2seg
        else:
            return hazy, clear

    def __len__(self):
        return len(self.hazy_paths)

    def get_img_name(self):
        return self.hazy_path, self.clear_path


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=240, format='.png', normalize=True, return_seg=False):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.normalize = normalize
        self.return_seg = return_seg  # 'normalize' and 'return_seg' could be merged as one paramter

        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        if train:
            self.clear_dir = os.path.join(path, 'clear')
        else:
            self.clear_dir = os.path.join(path, 'gt')

    def __getitem__(self, index):
        self.hazy_path = self.haze_imgs[index]
        haze = Image.open(self.hazy_path)
        # if isinstance(self.size,int):
        # while haze.size[0]<self.size or haze.size[1]<self.size:
        # index=random.randint(0,20000)
        # haze=Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]
        clear_name = id + self.format
        self.clear_path = os.path.join(self.clear_dir, clear_name)
        clear = Image.open(self.clear_path)
        # clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        # if args.return_seg:
            # haze, clear, haze2seg = self.augData(haze.convert("RGB"), clear.convert("RGB"))
            # return haze, clear, haze2seg
        # else: 
            # haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
            # return haze, clear
        return self.augData(haze.convert("RGB"), clear.convert("RGB"))

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        # if self.normalize:
            # data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        if self.return_seg:
            hazy2seg = tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        if self.return_seg:
            return data, target, hazy2seg
        else:
            return data, target

    def __len__(self):
        return len(self.haze_imgs)

    def get_img_name(self):
        return self.hazy_path, self.clear_path


if __name__ == '__main__':
    crop_h = 240
    crop_w = 320

    # dir_path = "../../../Dataset/NH-HAZE"
    # train_dataset = NH_HAZE(dir_path, mode='train', crop=True, crop_h=crop_h, crop_w=crop_w)
    # test_dataset = NH_HAZE(dir_path, mode='test')
    dir_path = "../../../Dataset/DENSE-HAZE"
    train_dataset = DENSE_HAZE(dir_path, mode='train', crop=True, crop_h=crop_h, crop_w=crop_w)
    test_dataset = DENSE_HAZE(dir_path, mode='test')

    train_hazy, train_clear = train_dataset[3]
    print(train_hazy.size(), train_clear.size())
    print(train_dataset.get_img_name())
    test_hazy, test_clear = test_dataset[3]
    print(test_hazy.size(), test_clear.size())
    print(test_dataset.get_img_name())

    crop_size = 240
    dir_path = "../../../Dataset/RESIDE"
    # train_dataset = RESIDE_Dataset(dir_path+'/Standard/ITS', train=True, size=crop_size)
    # test_dataset = RESIDE_Dataset(dir_path + '/Standard/SOTS/indoor', train=False, size="whole img")
    train_dataset = RESIDE_Dataset(dir_path + '/Beta/OTS', train=True, size=crop_size, format='.jpg')
    test_dataset = RESIDE_Dataset(dir_path + '/Standard/SOTS/indoor', train=False, size="whole img")
    train_hazy, train_clear = train_dataset[3]
    print(len(train_dataset))
    print(train_hazy.size(), train_clear.size())
    print(train_dataset.get_img_name())
    test_hazy, test_clear = test_dataset[3]
    print(len(test_dataset))
    print(test_hazy.size(), test_clear.size())
    print(test_dataset.get_img_name())

