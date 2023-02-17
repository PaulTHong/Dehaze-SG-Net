import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tfs
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
import argparse
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from PIL import Image
import glob

from models import *
from resnet import new_rf_lw50, new_rf_lw101, new_rf_lw152
from utils import tensor_normalize

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
    # dehaze_net = FFA(gps=3, blocks=19).cuda()
    seg_ckpt_path = '../../semantic_segmentation/light-weight-refinenet/ckpt/50_nyud.ckpt'
    dehaze_net = seg_att_FFA(gps=3, blocks=19, num_classes=40).cuda()
    refinenet = new_rf_lw50(num_classes=40, pretrain=True, ckpt_path=seg_ckpt_path).cuda()

    dehaze_net = torch.nn.DataParallel(dehaze_net)
    refinenet = torch.nn.DataParallel(refinenet)
    dehaze_net.load_state_dict(torch.load(args.model_path)['model'])
    dehaze_net.eval()
    refinenet.eval()

    mean = [0.64, 0.6, 0.58]
    std = [0.14, 0.15, 0.152]
    seg_mean = [0.485, 0.456, 0.406]
    seg_std = [0.229, 0.224, 0.225]

    print('\nStart Test!')
    total_psnr = 0
    total_ssim = 0
    for ite, (img_gt, img_haze) in enumerate(test_list):
        img_name = img_gt
        img_gt = Image.open(img_gt)
        img_haze = Image.open(img_haze)
        img_gt = np.array(img_gt)
        img_haze = np.array(img_haze)
        img_haze = tfs.ToTensor()(img_haze)[None, ::].cuda()

        haze1 = tfs.Normalize(mean=mean, std=std)(img_haze)
        haze2 = tfs.Normalize(mean=seg_mean, std=seg_std)(img_haze)
        with torch.no_grad():
            img_seg = refinenet(haze2)
            img_clean = dehaze_net(haze1, img_seg)
        
        img_gt = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0  

        img_clean = img_clean.clamp_(0, 1) 
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
    parser.add_argument('--model_path', type=str, default='./trained_models/its_train_ffa_3_19.pk')
    parser.add_argument('--test_output_folder', type=str, default="./test_results/")
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.test_output_folder):
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




