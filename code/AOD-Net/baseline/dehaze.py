import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import glob
from common import config

def dehaze_piece_image(dehaze_net, data_haze, clean_image, i1, i2, j1, j2, pad):
    haze_piece = data_haze[:, :, i1:i2+2*pad, j1:j2+2*pad]
    pred_piece = dehaze_net(haze_piece).cpu().detach().numpy()[0]
    if pad == 0:
        clean_image[:, i1:i2, j1:j2] = pred_piece[:, :, :]
    else:
        clean_image[:, i1:i2, j1:j2] = pred_piece[:, pad:-pad, pad:-pad]
    return clean_image


def crop_splice_image(dehaze_net, data_haze, args):
    '''
    crop image into pieces, test, then splicei(拼接) pieces
    Input:
        data_haze: NCHW cuda() 0~1
    Output:
        clean_image: CHW numpy() 0~1
    '''
    h_piece = args.h_piece
    w_piece = args.w_piece
    pad = args.pad
    
    H, W = data_haze.size()[2], data_haze.size()[3]
    if pad > 0:
        pad_haze = torch.zeros(1, 3, H+2*pad, W+2*pad)
        pad_haze[:, :, pad:-pad, pad:-pad] = data_haze
        data_haze = pad_haze.cuda()
    
    clean_image = np.zeros((3, H, W))
    h_count = H // h_piece
    w_count = W // w_piece
    h_left = H % h_piece
    w_left = W % w_piece
    for i in range(h_count):
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, (i+1)*h_piece, j*w_piece, (j+1)*w_piece, pad)
        if w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, (i+1)*h_piece, j*w_piece, W+1, pad)
    if h_left > 0:
        i = h_count
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, H+1, j*w_piece, (j+1)*w_piece, pad)
        if w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, H+1, j*w_piece, W+1, pad)
    return clean_image


def post_process(image):
    ## Contrast Augmentation
    for k in range(image.shape[-1]):
        min_pixel = np.min(image[:, :, k])
        max_pixel = np.max(image[:, :, k])
        image[:, :, k] = (image[:, :, k] - min_pixel) / (max_pixel - min_pixel)
    factor = 0.5
    image = np.clip(image/factor, 0, 1) 
    return image
    
    ## Histogram Equalization  First -> 0~255
    # for k in range(image.shape[-1]):
        # if k == 0:
            # res = cv2.equalizeHist(image[:, :, k])[:, :, np.newaxis]
        # else:
            # temp = cv2.equalizeHist(image[:, :, k])[:, :, np.newaxis]
            # res = np.concatenate((res, temp), axis=2)
    # return res

def dehaze_image(image_path, args): 
    data_haze = cv2.imread(image_path) / 255.0
    
    data_haze = torch.from_numpy(data_haze).float()
    data_haze = data_haze.permute(2, 0, 1)
    data_haze = data_haze.cuda().unsqueeze(0)

    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load(args.model_path))
    dehaze_net.eval()
    
    with torch.no_grad():
        if not args.crop:
            clean_image = dehaze_net(data_haze)
            clean_image = clean_image.cpu().detach().numpy()[0] 
            save_path = os.path.join(args.output_dir, os.path.split(image_path)[-1][:-4] + '.png')
        else:
            clean_image = crop_splice_image(dehaze_net, data_haze, args)
            save_path = os.path.join(args.output_dir, 'smooth' + str(args.pad) + '_' + os.path.split(image_path)[-1][:-4] + '.png')
            
    clean_image = clean_image.transpose((1, 2, 0)) 
    if args.post_process:
        print('Post-process: Contrast Augmentation.')
        clean_image = post_process(clean_image)
    else:
        clean_image = np.clip(clean_image, 0, 1)
    show_clean_image = np.uint8(clean_image * 255)
    # show_clean_image = clean_image * 255
    cv2.imwrite(save_path, show_clean_image)
    # torchvision.utils.save_image(torch.cat((data_hazy, show_clean_image), 0), save_path)  # transpose((0, 3, 1, 2)) ==>  BGR->RGB


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="../test_images")
    parser.add_argument('--model_path', type=str, default='./train_logs/snapshots/dehazer.pth')
    parser.add_argument('--output_dir', type=str, default='./demo_results')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--h_piece', type=int, default=config.height)
    parser.add_argument('--w_piece', type=int, default=config.width)
    parser.add_argument('--pad', type=int, default=3)
    parser.add_argument('--post_process', action='store_true', default=False)
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    test_list = glob.glob(os.path.join(args.test_dir, '*'))
    test_list = glob.glob(os.path.join(args.test_dir, '*'))
    for image in test_list:
        if os.path.isdir(image):
            continue
        t1 = time.time()
        dehaze_image(image, args)
        print(os.path.split(image)[-1], "done!")
        print('Time used:%.2fs' % (time.time()-t1))
