import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
import argparse
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import cv2
import glob

from GCANet import *
from common import config
from utils import edge_compute, tensor_normalize
from resnet import new_rf_lw50, new_rf_lw101, new_rf_lw152


def populate_test_list(gt_images_path, hazy_images_path):
    test_list = []
    image_list_haze = glob.glob(hazy_images_path + '*.jpg')
    image_list_haze.extend(glob.glob(hazy_images_path + '*.png'))
    tmp_dict = {}
    for image in image_list_haze:
        image = image.split('/')[-1]
        key = image.split('_')[0] + '.png'  # png maybe need be changed to jpg
        if key in tmp_dict.keys():
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

    for key in tmp_dict.keys():
        for hazy_image in tmp_dict[key]:
            test_list.append([gt_images_path + key, hazy_images_path + hazy_image])
    return test_list


def eval_index(gt_img, dehaze_img):
    gt_img = gt_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    dehaze_img = dehaze_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    psnr = peak_signal_noise_ratio(gt_img[0], dehaze_img[0])
    ssim = structural_similarity(gt_img[0], dehaze_img[0], multichannel=True, data_range=1)
    return psnr, ssim


def test(test_list, args):
    dehaze_net = seg_attention_GCANet(in_c=4, out_c=3, only_residual=args.only_residual,
                                      num_classes=args.seg_class_num).cuda()
    dehaze_net.load_state_dict(torch.load(args.model_path))
    refinenet = eval('new_rf_lw' + args.seg_model_res)(num_classes=args.seg_class_num, 
                                                       pretrain=True, ckpt_path=args.seg_ckpt_path).cuda()

    dehaze_net.eval()
    refinenet.eval()

    print('\nStart Test!')
    total_psnr = 0
    total_ssim = 0
    for ite, (img_gt, img_haze) in enumerate(test_list):
        img_name = img_gt
        img_gt = cv2.imread(img_gt)
        img_haze = cv2.imread(img_haze)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)

        # If the following steps are omitted, the sizes of img_clean and img_haze may be inconsistent
        # (1 pixel difference), then the addition operation will make error
        im_h, im_w = img_gt.shape[:2]
        if im_w % 4 != 0 or im_h % 4 != 0:
            img_gt = cv2.resize(img_gt, (int(im_w // 4 * 4), int(im_h // 4 * 4)))
            img_haze = cv2.resize(img_haze, (int(im_w // 4 * 4), int(im_h // 4 * 4)))
        if args.resize:
            img_gt = cv2.resize(img_gt, (config.height, config.width), cv2.INTER_LINEAR)
            img_haze = cv2.resize(img_haze, (config.height, config.width), cv2.INTER_LINEAR)
        img_gt = torch.from_numpy(img_gt.transpose((2, 0, 1))).float()
        img_haze = torch.from_numpy(img_haze.transpose((2, 0, 1))).float()
        
        norm_mean = (0.485, 0.456, 0.406) 
        norm_std = (0.229, 0.224, 0.255)
        with torch.no_grad():
            normalized_haze = tensor_normalize(img_haze / 255.0, norm_mean, norm_std).cuda()
            img_seg = refinenet(normalized_haze)
        
        if args.add_edge:
            edge = edge_compute(img_haze)
            img_in = torch.cat((img_haze, edge), dim=0) - 128
        else:
            img_in = img_haze - 128

        img_gt = img_gt.unsqueeze(0).cuda() / 255.0  
        img_haze = img_haze.unsqueeze(0).cuda() 
        img_in = img_in.unsqueeze(0).cuda()

        with torch.no_grad():
            img_out = dehaze_net(img_in, img_seg)
        if args.only_residual:
            img_clean = img_out + img_haze
        else:
            img_clean = img_out

        img_clean = img_clean.round().clamp_(0, 255) / 255.0
        img_haze = img_haze / 255.0  
        psnr, ssim = eval_index(img_gt, img_clean)
        total_psnr += psnr
        total_ssim += ssim
        print('iter %d: ' % ite, os.path.split(img_name)[-1], ' PSNR: %.4f  SSIM %.4f' % (psnr, ssim))

        if args.save_image:
            torchvision.utils.save_image(
                torch.cat((img_haze, img_clean, img_gt), 0),
                args.test_output_folder + os.path.split(img_name)[-1])

    num = len(test_list)
    total_psnr = total_psnr / num
    total_ssim = total_ssim / num
    print('Test numbers: %d' % num)
    print('Test Average PSNR: %.4f' % total_psnr)
    print('Test Average SSIM: %.4f' % total_ssim)

    return total_psnr, total_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./train_logs/models/dehaze.ckpt')
    parser.add_argument('--test_output_folder', type=str, default="./test_results/")
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--add_edge', action='store_true')
    parser.add_argument('--only_residual', action='store_true', default=False)
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--seg_ckpt_path', type=str, default='../../semantic_segmentation/light-weight-refinenet/ckpt/50_nyud.ckpt')
    parser.add_argument('--seg_model_res', type=str, default='50', choices=['50', '101', '152'])
    parser.add_argument('--seg_dataset', type=str, default='nyud', choices=['nyud', 'voc'])
    parser.add_argument('--seg_class_num', type=int, default=40)

    args = parser.parse_args()
    args.seg_ckpt_path = '../../semantic_segmentation/light-weight-refinenet/ckpt/' + \
            args.seg_model_res + '_' + args.seg_dataset + '.ckpt'
    for k, v in args.__dict__.items():
        print(k, ': ', v)

    if args.save_image and not os.path.exists(args.test_output_folder):
        os.system('mkdir -p %s' % args.test_output_folder)

    t1 = time.time()
    indoor_gt = '../../../Dataset/RESIDE/Standard/SOTS/indoor/gt/'
    indoor_hazy = '../../../Dataset/RESIDE/Standard/SOTS/indoor/hazy/'
    indoor_list = populate_test_list(indoor_gt, indoor_hazy)
    indoor_num = len(indoor_list)
    indoor_psnr, indoor_ssim = test(indoor_list, args)
    indoor_time = time.time() - t1

    t2 = time.time()
    outdoor_gt = '../../../Dataset/RESIDE/Standard/SOTS/outdoor/gt/'
    outdoor_hazy = '../../../Dataset/RESIDE/Standard/SOTS/outdoor/hazy/'
    outdoor_list = populate_test_list(outdoor_gt, outdoor_hazy)
    outdoor_num = len(outdoor_list)
    outdoor_psnr, outdoor_ssim = test(outdoor_list, args)
    outdoor_time = time.time() - t2

    print('\nIndoor time consume: %.2f' % indoor_time)
    print('Indoor test numbers: %d' % indoor_num)
    print('Indoor Average PSNR: %.4f' % indoor_psnr)
    print('Indoor Average SSIM: %.4f' % indoor_ssim)

    print('\nOutdoor time consume:%.2f' % outdoor_time)
    print('Outdoor test numbers: %d' % outdoor_num)
    print('Outdoor Average PSNR: %.4f' % outdoor_psnr)
    print('Outdoor Average SSIM: %.4f' % outdoor_ssim)

    total_psnr = (indoor_num * indoor_psnr + outdoor_num * outdoor_psnr) / (indoor_num + outdoor_num)
    total_ssim = (indoor_num * indoor_ssim + outdoor_num * outdoor_ssim) / (indoor_num + outdoor_num)
    print('\nTotal Test Average PSNR: %.4f' % total_psnr)
    print('Total Test Average SSIM: %.4f' % total_ssim)
    print('Test over!')




