import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import math
import numpy as np
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
import cv2

from dataloader import NH_HAZE, DENSE_HAZE, RESIDE_Dataset
from AECRNet import Dehaze, SGDehaze, SFDehaze
from CR import ContrastLoss, JointCRLoss
from resnet import new_rf_lw50, new_rf_lw101, new_rf_lw152


def eval_index(img_gt, img_pred):
    psnr = peak_signal_noise_ratio(img_gt, img_pred)
    ssim = structural_similarity(img_gt, img_pred, multichannel=True, data_range=1)
    return psnr, ssim


# Consider: crop_h + 2*pad = orig_crop_h, crop_w + 2*pad = orig_crop_w
def dehaze_piece_image(dehaze_net, data_hazy, img_pred, i1, i2, j1, j2, pad, refinenet=None, data_hazy2seg=None):
    if data_hazy2seg is not None:
        hazy2seg_piece = data_hazy2seg[:, :, i1:i2 + 2 * pad, j1:j2 + 2 * pad]

    hazy_piece = data_hazy[:, :, i1:i2+2*pad, j1:j2+2*pad]
    if refinenet:
        seg_piece = refinenet(hazy2seg_piece)
        pred_piece = dehaze_net(hazy_piece, seg_piece)
    else:
        pred_piece = dehaze_net(hazy_piece)
    if pad == 0:
        img_pred[:, :, i1:i2, j1:j2] = pred_piece
    else:
        img_pred[:, :, i1:i2, j1:j2] = pred_piece[:, :, pad:-pad, pad:-pad]
    return img_pred


def crop_splice_image(dehaze_net, data_hazy, args, refinenet=None, data_hazy2seg=None):
    '''
    Crop image into pieces, test, then splice pieces.
    Input:
        data_haze: NCHW cuda() 0~1
    Output:
        img_pred: NCHW cuda() 0~1   # CHW numpy() 0~1
    '''
    h_piece = args.crop_h
    w_piece = args.crop_w
    pad = args.pad

    H, W = data_hazy.size()[2], data_hazy.size()[3]
    if pad > 0:
        pad_hazy = torch.zeros(1, 3, H + 2 * pad, W + 2 * pad)
        pad_hazy[:, :, pad:-pad, pad:-pad] = data_hazy
        data_hazy = pad_hazy.cuda()
        if data_hazy2seg is not None:
            pad_hazy2seg = torch.zeros(1, 3, H + 2 * pad, W + 2 * pad)
            pad_hazy2seg[:, :, pad:-pad, pad:-pad] = data_hazy2seg
            data_hazy2seg = pad_hazy2seg.cuda()

    img_pred = torch.zeros(1, 3, H, W)
    h_count = H // h_piece
    w_count = W // w_piece
    h_left = H % h_piece
    w_left = W % w_piece
    for i in range(h_count):
        for j in range(w_count):
            img_pred = dehaze_piece_image(
                dehaze_net, data_hazy, img_pred, i * h_piece, (i + 1) * h_piece,
                j * w_piece, (j + 1) * w_piece, pad, refinenet, data_hazy2seg)
        if w_left > 0:
            j = w_count
            img_pred = dehaze_piece_image(
                dehaze_net, data_hazy, img_pred, i * h_piece, (i + 1) * h_piece,
                j * w_piece, W + 1, pad, refinenet, data_hazy2seg)
    if h_left > 0:
        i = h_count
        for j in range(w_count):
            img_pred = dehaze_piece_image(
                dehaze_net, data_hazy, img_pred, i * h_piece, H + 1,
                j * w_piece, (j + 1) * w_piece, pad, refinenet, data_hazy2seg)
        if w_left > 0:
            j = w_count
            img_pred = dehaze_piece_image(
                dehaze_net, data_hazy, img_pred, i * h_piece, H + 1,
                j * w_piece, W + 1, pad, refinenet, data_hazy2seg)
    return img_pred


def test(args, dehaze_net, test_dataset, refinenet=None):
    print('\nStart Test!')

    dehaze_net.eval()
    if refinenet:
        refinenet.eval()
    total_psnr = 0
    total_ssim = 0
    for iter_test, data in enumerate(test_dataset):
        if args.return_seg:
            img_hazy, img_gt, img_hazy2seg = data
            img_hazy, img_gt, img_hazy2seg = img_hazy.unsqueeze(0), img_gt.unsqueeze(0), img_hazy2seg.unsqueeze(0)
            img_hazy = img_hazy.cuda()
            img_gt = img_gt.cuda()
            img_hazy2seg = img_hazy2seg.cuda()
            with torch.no_grad():
                if args.crop:
                    img_pred = crop_splice_image(dehaze_net, img_hazy, args, refinenet, img_hazy2seg)
                else:
                    img_seg = refinenet(img_hazy2seg)
                    img_pred = dehaze_net(img_hazy, img_seg)
                img_pred = torch.clamp(img_pred, 0, 1)
        else:
            img_hazy, img_gt = data
            img_hazy, img_gt = img_hazy.unsqueeze(0), img_gt.unsqueeze(0)
            img_hazy = img_hazy.cuda()
            img_gt = img_gt.cuda()
            with torch.no_grad():
                if args.crop:
                    img_pred = crop_splice_image(dehaze_net, img_hazy, args)
                else:
                    img_pred = dehaze_net(img_hazy)
                img_pred = torch.clamp(img_pred, 0, 1)

        img_gt = img_gt.cpu().detach().numpy().transpose((0, 2, 3, 1))[0]
        img_pred = img_pred.cpu().detach().numpy().transpose((0, 2, 3, 1))[0]
        if args.resize:
            img_pred = cv2.resize(img_pred, (args.ORIG_WIDTH, args.ORIG_HEIGHT), cv2.INTER_LINEAR)

        psnr, ssim = eval_index(img_gt, img_pred)
        total_psnr += psnr
        total_ssim += ssim
        print('Batch %d - Test PSNR: %.4f' % (iter_test+1, psnr))
        print('Batch %d - Test SSIM: %.4f' % (iter_test+1, ssim))

        if args.test_save:
            hazy_path, gt_path = test_dataset.get_img_name()
            save_path = 'pred_' + os.path.split(hazy_path)[-1]
            print(save_path)
            save_path = os.path.join(args.test_output_folder, save_path)
            img_pred = np.uint8(img_pred * 255)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_pred)

    total_psnr = total_psnr / args.test_total_num * args.test_batch_size
    total_ssim = total_ssim / args.test_total_num * args.test_batch_size
    print('\nAverage test PSNR: %.4f' % total_psnr)
    print('Average test SSIM: %.4f' % total_ssim)
    return total_psnr, total_ssim


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--data_name', type=str, default='ITS', choices=['ITS', 'OTS', 'NH-HAZE', 'DENSE-HAZE'])
    parser.add_argument('--dir_path', type=str, default="../../../Dataset/RESIDE")
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--crop_size', type=int, default=240)
    parser.add_argument('--crop_h', type=int, default=240)
    parser.add_argument('--crop_w', type=int, default=320)
    parser.add_argument('--pad', type=int, default=10)
    parser.add_argument('--resize', action='store_true',
                        help='Resize as the size of training before test, and resize back after test.')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='./train_logs/best_dehazer.pth')
    parser.add_argument('--test_save', action='store_true')
    parser.add_argument('--test_output_folder', type=str, default='./test_results')
    parser.add_argument('--multi_gpu', action='store_true')

    parser.add_argument('--model_mode', type=str, default='SG', choices=['base', 'SG'])
    parser.add_argument('--seg_ckpt_path', type=str,
                        default='../../semantic_segmentation/light-weight-refinenet/ckpt/50_nyud.ckpt')
    parser.add_argument('--seg_model_res', type=str, default='50', choices=['50', '101', '152'])
    parser.add_argument('--seg_dataset', type=str, default='nyud', choices=['nyud', 'voc'])
    parser.add_argument('--seg_class_num', type=int, default=40)  # nyud: 40, voc: 21
    parser.add_argument('--seg_dim', type=int, default=16) 
    args = parser.parse_args()
    args.return_seg = args.model_mode != 'base'
    rs = args.return_seg

    if args.test_save and not os.path.exists(args.test_output_folder):
        os.system('mkdir -p %s' % args.test_output_folder)

    t1 = time.time()

    if args.data_name == 'NH-HAZE':
        args.dir_path = "../../../Dataset/NH-HAZE"
        args.ORIG_HEIGHT, args.ORIG_WIDTH = 1200, 1600
        test_dataset = NH_HAZE(args.dir_path, mode='test',
                               crop_h=args.crop_h, crop_w=args.crop_w, resize=args.resize, return_seg=rs)
    elif args.data_name == 'DENSE-HAZE':
        args.dir_path = "../../../Dataset/DENSE-HAZE"
        args.ORIG_HEIGHT, args.ORIG_WIDTH = 1200, 1600
        test_dataset = DENSE_HAZE(args.dir_path, mode='test',
                                  crop_h=args.crop_h, crop_w=args.crop_w, resize=args.resize, return_seg=rs)
    elif args.data_name == 'ITS':
        args.dir_path = "../../../Dataset/RESIDE"
        args.ORIG_HEIGHT, args.ORIG_WIDTH = 460, 620
        test_dataset = RESIDE_Dataset(args.dir_path + '/Standard/SOTS/indoor',
                                      train=False, size='whole img', return_seg=rs)
    elif args.data_name == 'OTS':
        args.dir_path = "../../../Dataset/RESIDE"
        args.ORIG_HEIGHT, args.ORIG_WIDTH = 460, 620
        test_dataset = RESIDE_Dataset(args.dir_path + '/Standard/SOTS/indoor',
                                      train=False, size="whole img", return_seg=rs)
    
    for k, v in args.__dict__.items():
        print(k, ': ', v)

    print('\nData loaded. Test num: %d.' % (len(test_dataset)))
    args.test_total_num = len(test_dataset)

    if args.model_mode == 'base':
        dehaze_net = Dehaze(input_nc=3, output_nc=3).cuda()
    elif args.model_mode == 'SG':
        dehaze_net = SGDehaze(input_nc=3, output_nc=3, seg_dim=args.seg_dim, num_classes=args.seg_class_num).cuda()
    else:
        raise NotImplementedError
    
    dehaze_net.load_state_dict(torch.load(args.model_path))

    if args.model_mode == 'base':
        refinenet = None
    else:
        refinenet = eval('new_rf_lw'+args.seg_model_res)(
            num_classes=args.seg_class_num, pretrain=True, ckpt_path=args.seg_ckpt_path).cuda()
        for p in refinenet.parameters():
            p.requires_grad = False
        refinenet.eval()

    if args.multi_gpu:
        dehaze_net = nn.DataParallel(dehaze_net)
        if refinenet:
            refinenet = nn.DataParallel(refinenet)

    test(args, dehaze_net, test_dataset, refinenet)
    print('Time consume: %.2f min' % ((time.time() - t1) / 60))
