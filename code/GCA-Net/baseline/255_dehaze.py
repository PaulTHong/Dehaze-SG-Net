import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
import argparse
import time
import cv2
import glob

from common import config
from GCANet import GCANet
from utils import edge_compute


def dehaze_piece_image(dehaze_net, data_haze, data_in, clean_image, i1, i2, j1, j2, pad):
    haze_piece = data_haze[:, :, i1:i2 + 2 * pad, j1:j2 + 2 * pad]
    in_piece = data_in[:, :, i1:i2 + 2 * pad, j1:j2 + 2 * pad]
    pred_piece = dehaze_net(in_piece)
    if args.only_residual:
        pred_piece = pred_piece + haze_piece
    pred_piece = pred_piece.cpu().detach().numpy()[0]
    if pad == 0:
        clean_image[:, i1:i2, j1:j2] = pred_piece[:, :, :]
    else:
        clean_image[:, i1:i2, j1:j2] = pred_piece[:, pad:-pad, pad:-pad]
    return clean_image


def crop_splice_image(dehaze_net, data_haze, data_in, args):
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
        pad_haze = torch.zeros(1, 3, H + 2 * pad, W + 2 * pad)
        pad_haze[:, :, pad:-pad, pad:-pad] = data_haze
        data_haze = pad_haze.cuda()
        in_c = 4 if args.add_edge else 3
        pad_in = torch.zeros(1, in_c, H + 2 * pad, W + 2 * pad)
        pad_in[:, :, pad:-pad, pad:-pad] = data_in
        data_in = pad_in.cuda()

    clean_image = np.zeros((3, H, W))
    h_count = H // h_piece
    w_count = W // w_piece
    h_left = H % h_piece
    w_left = W % w_piece
    for i in range(h_count):
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, data_in, clean_image, i * h_piece,
                                             (i + 1) * h_piece, j * w_piece, (j + 1) * w_piece, pad)
        if w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, data_in, clean_image, i * h_piece,
                                             (i + 1) * h_piece, j * w_piece, W + 1, pad)
    if h_left > 0:
        i = h_count
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, data_in, clean_image, i * h_piece,
                                             H + 1, j * w_piece, (j + 1) * w_piece, pad)
        if w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, data_in, clean_image, i * h_piece,
                                             H + 1, j * w_piece, W + 1, pad)
    return clean_image


def post_process(image):
    pass

def dehaze_image(image_path, args):
    in_c = 4 if args.add_edge else 3 
    dehaze_net = GCANet(in_c=in_c, out_c=3, only_residual=args.only_residual).cuda()
    dehaze_net.load_state_dict(torch.load(args.model_path))
    dehaze_net.eval()

    img_haze = cv2.imread(image_path)
    im_h, im_w = img_haze.shape[:2]
    if im_w % 4 != 0 or im_h % 4 != 0:
        img_haze = cv2.resize(img_haze, (int(im_w // 4 * 4), int(im_h // 4 * 4)))
    img_haze = torch.from_numpy(img_haze.transpose((2, 0, 1))).float()
    if args.add_edge:
        edge = edge_compute(img_haze)
        img_in = torch.cat((img_haze, edge), dim=0) - 128
    else:
        img_in = img_haze - 128
    img_haze = img_haze.cuda().unsqueeze(0)
    img_in = img_in.cuda().unsqueeze(0)

    with torch.no_grad():
        if not args.crop:
            img_clean = dehaze_net(img_in)
            if args.only_residual:
                img_clean = (img_clean + img_haze).round().clamp_(0, 255)
            img_clean = img_clean.cpu().detach().numpy()[0]
            save_path = os.path.join(args.output_dir, os.path.split(image_path)[-1][:-4] + '.png')
        else:
            img_clean = crop_splice_image(dehaze_net, img_haze, img_in, args).round().clip(0, 255)
            save_path = os.path.join(args.output_dir,
                                     'smooth' + str(args.pad) + '_' + os.path.split(image_path)[-1][:-4] + '.png')

    img_clean = img_clean.transpose((1, 2, 0))
    if args.post_process:
        print('Post-process: Contrast Augmentation.')
        img_clean = post_process(img_clean / 255.0) * 255.0
    else:
        pass
    show_clean_image = np.uint8(img_clean)
    cv2.imwrite(save_path, show_clean_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="../demo_images")
    parser.add_argument('--model_path', type=str, default='./train_logs/models/best_dehaze.ckpt')
    parser.add_argument('--output_dir', type=str, default='./demo_results')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--h_piece', type=int, default=config.height)
    parser.add_argument('--w_piece', type=int, default=config.width)
    parser.add_argument('--pad', type=int, default=3)
    parser.add_argument('--post_process', action='store_true', default=False)
    parser.add_argument('--add_edge', action='store_true')
    parser.add_argument('--only_residual', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    test_list = glob.glob(os.path.join(args.test_dir, '*'))
    for image in test_list:
        if os.path.isdir(image):
            continue
        t1 = time.time()
        dehaze_image(image, args)
        print(os.path.split(image)[-1], "done!")
        print('Time used:%.2fs' % (time.time() - t1))
