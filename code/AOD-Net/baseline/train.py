import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import numpy as np
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import dataloader
import net
from common import config

def weights_init(m):
    if isinstance(m, nn.Conv2d):  # proposed way
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def eval_index(gt_img, dehaze_img):
    gt_img = gt_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    dehaze_img = dehaze_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    psnr, ssim = 0, 0
    N = gt_img.shape[0]
    for i in range(N):
        psnr += peak_signal_noise_ratio(gt_img[i], dehaze_img[i])
        ssim += structural_similarity(gt_img[i], dehaze_img[i], multichannel=True, data_range=1)
    return psnr/N, ssim/N


def train(args, dehaze_net):
    if args.model_trained:
        dehaze_net.load_state_dict(torch.load(args.model_path))
        print('\nModel loaded without train!')
    else:
        print('\nStart train!')
        dehaze_net.apply(weights_init)

        train_dataset = dataloader.dehazing_loader(args.orig_images_path,
                                                 args.hazy_images_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                            shuffle=True, num_workers=args.num_workers, pin_memory=True)

        if args.loss_func == 'l1':
            criterion = nn.SmoothL1Loss().cuda()
        elif args.loss_func == 'l2':
            criterion = nn.MSELoss().cuda()
        else:
            print('loss_func %s not supported' % args.loss_func)
        optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        dehaze_net.train()

        for epoch in range(args.num_epochs):
            print('lr:%.6f' % schedule.get_lr()[0])
            total_loss = 0
            for iteration, (img_orig, img_haze) in enumerate(train_loader):
                img_orig = img_orig.cuda()
                img_haze = img_haze.cuda()

                clean_image = dehaze_net(img_haze)
                loss = criterion(clean_image, img_orig)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), args.grad_clip_norm)
                optimizer.step()

                if ((iteration+1) % args.display_iter) == 0:
                    print("Loss at epoch", epoch+1, "| iteration", iteration+1, ":%.8f" % loss.item())
                total_loss += loss.item()
                if ((iteration+1) % args.snapshot_iter) == 0:
                    save_model = dehaze_net.module.state_dict() if args.multi_gpu else dehaze_net.state_dict()
                    torch.save(save_model, args.snapshots_folder + "Epoch" + str(epoch) + '.pth')
            print('Average loss at epoch', epoch+1, ":%.8f" % (total_loss/len(train_loader)))
            schedule.step()  # adjust learning rate
            save_model = dehaze_net.module.state_dict() if args.multi_gpu else dehaze_net.state_dict()
            torch.save(save_model, args.snapshots_folder + "dehazer.pth")
    
    # valid(args, dehaze_net)

# Validation Stage
def valid(args, dehaze_net):
    print('\nStart validation!')
    val_dataset = dataloader.dehazing_loader(args.orig_images_path,
                                            args.hazy_images_path, mode="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)  # shuffle=True

    dehaze_net.eval()
    total_psnr = 0
    total_ssim = 0
    for iter_val, (img_orig, img_haze) in enumerate(val_loader):
        img_orig = img_orig.cuda()
        img_haze = img_haze.cuda()
        with torch.no_grad():
            clean_image = dehaze_net(img_haze)
            clean_image = torch.clamp(clean_image, 0, 1)

        psnr, ssim = eval_index(img_orig, clean_image)
        total_psnr += psnr 
        total_ssim += ssim
        print('Batch %d - Validate PSNR: %.4f' % (iter_val, psnr))
        print('Batch %d - Validate SSIM: %.4f' % (iter_val, ssim))
        
        # permute [2,1,0] means convert BGR to RGB to display by torchvision
        if not args.valid_not_save:
            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         args.sample_output_folder+str(iter_val+1)+".jpg")

    total_psnr = total_psnr / len(val_dataset) * args.val_batch_size
    total_ssim = total_ssim / len(val_dataset) * args.val_batch_size
    print('Validate PSNR: %.4f' % total_psnr)
    print('Validate SSIM: %.4f' % total_ssim)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--dataset', type=str, default='ITS', choices=['ITS', 'OTS'])
    lr = parser.add_argument_group(title='Learning rate')
    lr.add_argument('--init_lr', type=float, default=0.0001)
    lr.add_argument('--milestones', nargs='+', type=int, default=[4, 7])
    lr.add_argument('--gamma', type=float, default=0.1)     
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--loss_func', default='l1', help='l1|l2')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="./train_logs/snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="./train_logs/samples/")
    parser.add_argument('--model_path', type=str, default='./train_logs/snapshots/dehazer.pth')
    parser.add_argument('--model_trained', action='store_true')
    parser.add_argument('--valid_not_save', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'ITS':
        args.orig_images_path="../../../Dataset/RESIDE/Standard/ITS/clear/"
        args.hazy_images_path="../../../Dataset/RESIDE/Standard/ITS/hazy/"
    elif args.dataset == 'OTS':
        args.orig_images_path="../../../Dataset/RESIDE/Beta/OTS/clear/"
        args.hazy_images_path="../../../Dataset/RESIDE/Beta/OTS/hazy/"

    for k, v in args.__dict__.items():
        print(k, ': ', v)
    
    if not os.path.exists(args.snapshots_folder):
        os.system('mkdir -p %s' % args.snapshots_folder)
    if not os.path.exists(args.sample_output_folder):
        os.system('mkdir -p %s' % args.sample_output_folder)

    t1 = time.time()
    dehaze_net = net.dehaze_net().cuda()
    if args.multi_gpu:
        dehaze_net = nn.DataParallel(dehaze_net)
    train(args, dehaze_net)
    valid(args, dehaze_net)
    print('Time consume: %.2f h' % ((time.time()-t1) / 3600))
