import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import os
import sys
import argparse
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import dataloader
from GCANet import GCANet
from common import config
from utils import edge_compute


def eval_index(gt_img, dehaze_img):
    gt_img = gt_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    dehaze_img = dehaze_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    psnr, ssim = 0, 0
    N = gt_img.shape[0]
    for i in range(N):
        psnr += peak_signal_noise_ratio(gt_img[i], dehaze_img[i])
        ssim += structural_similarity(gt_img[i], dehaze_img[i], multichannel=True, data_range=1)
    return psnr / N, ssim / N


def train(args, dehaze_net):
    if args.model_trained:
        dehaze_net.load_state_dict(torch.load(args.model_path))
        print('\nModel loaded without train!')
    else:
        print('\nStart train!')
        train_dataset = dataloader.dehazing_loader(args.orig_images_path,
                                                   args.hazy_images_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)

        if args.loss_func == 'l2':
            criterion = nn.MSELoss().cuda()
        elif args.loss_func == 'l1':
            criterion = nn.SmoothL1Loss().cuda()
        else:
            print('loss_func %s not supported' % args.loss_func)
            raise ValueError
        optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)  # last_epoch=-1

        if args.resume and os.path.isfile(args.model_path):
            print('==> Resuming from checkpoint..')
            dehaze_net.load_state_dict(torch.load(args.model_path))
            for _ in range(args.start_epoch):
                schedule.step()

        min_loss = float('inf')
        max_psnr = 0
        best_epoch = args.start_epoch  # 0
        for epoch in range(args.start_epoch, args.num_epochs):
            dehaze_net.train()
            print('lr:%.5f' % schedule.get_lr()[0])
            sum_loss = 0
            for iteration, (img_orig, img_haze) in enumerate(train_loader):
                img_orig = img_orig.cuda()
                img_haze = img_haze.cuda()
                if args.add_edge:
                    size = list(img_haze.size())
                    size[1] += 1
                    img_in = torch.zeros(*size).cuda()
                    for b in range(img_haze.size()[0]):
                        edge = edge_compute(img_haze[b])
                        img_in[b] = torch.cat((img_haze[b], edge), dim=0) - 128
                else:
                    img_in = img_haze - 128

                img_out = dehaze_net(img_in)
                if args.only_residual:
                    loss = criterion(img_out, img_orig - img_haze)
                else:
                    loss = criterion(img_out, img_orig)
                sum_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ((iteration + 1) % args.display_iter) == 0:
                    print("Loss at epoch", epoch + 1, "| iteration", iteration + 1, ":%.8f" % loss.item())
            schedule.step()  # adjust learning rate
            save_model = dehaze_net.module.state_dict() if args.multi_gpu else dehaze_net.state_dict() 
            torch.save(save_model, args.model_path)
            print("Average of loss at epoch", epoch + 1, ":%.8f" % (sum_loss.item() / len(train_loader)))
            if (epoch + 1) % 5 == 0:
                valid_psnr, valid_ssim = valid(args, dehaze_net)
                if valid_psnr > max_psnr:
                    max_psnr = valid_psnr
                    best_epoch = epoch + 1
                    torch.save(save_model, args.model_folder + "best_" + os.path.split(args.model_path)[-1])  # "best_dehaze.ckpt"
            if sum_loss < min_loss:
                min_loss = sum_loss
                print('best_epoch: %d, min_loss: %.4f, max_psnr: %.4f' % (best_epoch, min_loss, max_psnr))
        print('Train end!')


# Validation Stage
def valid(args, dehaze_net):
    print('\nStart validation!')
    val_dataset = dataloader.dehazing_loader(args.orig_images_path,
                                             args.hazy_images_path, mode="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)

    dehaze_net.eval()
    total_psnr = 0
    total_ssim = 0
    for iter_val, (img_orig, img_haze) in enumerate(val_loader):
        img_orig = img_orig.cuda()
        img_haze = img_haze.cuda()
        if args.add_edge:
            size = list(img_haze.size())
            size[1] += 1
            img_in = torch.zeros(*size).cuda()
            for b in range(img_haze.size()[0]):
                edge = edge_compute(img_haze[b])
                img_in[b] = torch.cat((img_haze[b], edge), dim=0) - 128
        else:
            img_in = img_haze - 128

        with torch.no_grad():
            img_out = dehaze_net(img_in)
        if args.only_residual:
            img_clean = img_out + img_haze
        else:
            img_clean = img_out

        img_orig = img_orig / 255.0  # ...
        img_clean = img_clean.round().clamp_(0, 255) / 255.0
        psnr, ssim = eval_index(img_orig, img_clean)
        total_psnr += psnr
        total_ssim += ssim
        if iter_val % 10 == 0:
            print('Batch %d - Validate PSNR: %.4f' % (iter_val, psnr))
            print('Batch %d - Validate SSIM: %.4f' % (iter_val, ssim))

        # permute [2,1,0] means convert BGR to RGB to display by torchvision
        if not args.valid_not_save:
            torchvision.utils.save_image(
                torch.cat((img_haze, img_clean, img_orig),
                          0), args.sample_output_folder + str(iter_val + 1) + ".jpg", nrow=args.val_batch_size)

    total_psnr = total_psnr / len(val_dataset) * args.val_batch_size
    total_ssim = total_ssim / len(val_dataset) * args.val_batch_size
    print('Validate PSNR: %.4f' % total_psnr)
    print('Validate SSIM: %.4f' % total_ssim)
    return total_psnr, total_ssim


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--dataset', type=str, default='ITS', choices=['ITS', 'OTS'])
    lr = parser.add_argument_group(title='Learning rate')
    lr.add_argument('--init_lr', type=float, default=0.001)
    lr.add_argument('--milestones', nargs='+', type=int, default=[40, 80])
    lr.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--loss_func', default='l2', help='l2|l1')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='index of epoch start from 0, but count from 1 when output, so set start_epoch as the epoch of the last output')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from pretrained checkpoint')
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--sample_output_folder', type=str, default="./train_logs/samples/")
    parser.add_argument('--model_path', type=str, default='./train_logs/models/dehaze.ckpt')
    parser.add_argument('--model_trained', action='store_true')
    parser.add_argument('--valid_not_save', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')

    parser.add_argument('--add_edge', action='store_true')
    parser.add_argument('--only_residual', action='store_true', default=False, help='regress residual rather than clean image')
    args = parser.parse_args()
    if args.dataset == 'ITS':
        args.orig_images_path = '../../../Dataset/RESIDE/Standard/ITS/clear/'
        args.hazy_images_path = '../../../Dataset/RESIDE/Standard/ITS/hazy/'
    elif args.dataset == 'OTS':
        args.orig_images_path = '../../../Dataset/RESIDE/Beta/OTS/clear/'
        args.hazy_images_path = '../../../Dataset/RESIDE/Beta/OTS/hazy/'
    print('Args namespace:')
    for k, v in args.__dict__.items():
        print(k, ': ', v)
    args.model_folder = os.path.split(args.model_path)[0] + '/'  # os.path.sep

    if not os.path.exists(args.model_folder):
        os.system('mkdir -p %s ' % args.model_folder)
    if not os.path.exists(args.sample_output_folder):
        os.system('mkdir -p %s' % args.sample_output_folder)

    t1 = time.time()
    in_c = 4 if args.add_edge else 3
    dehaze_net = GCANet(in_c=in_c, out_c=3, only_residual=args.only_residual).cuda()
    if args.multi_gpu:
        dehaze_net = nn.DataParallel(dehaze_net)
    train(args, dehaze_net)
    valid(args, dehaze_net)  # validation here is for the model of the last epoch
    print('Time consume: %.2f h' % ((time.time() - t1) / 3600))
