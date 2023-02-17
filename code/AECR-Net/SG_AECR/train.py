import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import warnings
import argparse
import time
import math
import numpy as np
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from dataloader import NH_HAZE, DENSE_HAZE, RESIDE_Dataset
from AECRNet import Dehaze, SGDehaze, SFDehaze
from CR import ContrastLoss, JointCRLoss
from resnet import new_rf_lw50, new_rf_lw101, new_rf_lw152

warnings.filterwarnings('ignore')

def lr_schedule_cosdecay(t, T, init_lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    # low_bound = init_lr * 0.15  # 0.15: 3/4 T, 0.25: 2/3 T 
    # lr = low_bound if lr < low_bound else lr
    return lr


def eval_index(img_gt, img_pred):
    img_gt = img_gt.cpu().detach().numpy().transpose((0, 2, 3, 1))
    img_pred = img_pred.cpu().detach().numpy().transpose((0, 2, 3, 1))
    psnr, ssim = 0, 0
    N = img_gt.shape[0]
    for i in range(N):
        psnr += peak_signal_noise_ratio(img_gt[i], img_pred[i])
        ssim += structural_similarity(img_gt[i], img_pred[i], multichannel=True, data_range=1)
    return psnr / N, ssim / N


def train(args, dehaze_net, train_loader, val_loader, refinenet=None):
    if args.model_trained:
        dehaze_net.load_state_dict(torch.load(args.model_path))
        print('\nModel loaded without train!')
    else:
        print('\nStart train!')
    
        criterion = JointCRLoss(contrast_w=args.contrast_w, neg_num=args.neg_num)
        optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, dehaze_net.parameters()),
                                     lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08)  # weight_decay=args.weight_decay
        optimizer.zero_grad()

        if args.resume and os.path.isfile(args.model_path):
            print('==> Resuming from checkpoint...')
            dehaze_net.load_state_dict(torch.load(args.model_path))

        best_psnr = 0
        best_ssim = 0
        best_epoch = 0
        for epoch in range(args.start_epoch, args.num_epochs):
            dehaze_net.train()
            lr = lr_schedule_cosdecay(epoch, args.num_epochs, args.init_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            if args.return_seg:
                (img_hazy, img_gt, img_haze2seg) = next(iter(train_loader))
                img_hazy = img_hazy.cuda()
                img_gt = img_gt.cuda()
                img_haze2seg = img_haze2seg.cuda()
                
                img_gt2seg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_gt)
                img_seg = refinenet(img_haze2seg)
                gt_seg = refinenet(img_gt2seg)
                
                img_pred = dehaze_net(img_hazy, img_seg) 
                img_pred2seg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_pred)
                pred_seg = refinenet(img_pred2seg)
            else:
                (img_hazy, img_gt) = next(iter(train_loader))
                img_hazy = img_hazy.cuda()
                img_gt = img_gt.cuda()

                img_pred = dehaze_net(img_hazy)

            loss = criterion(img_pred, img_gt, img_hazy)
            # loss_vgg7, all_ap, all_an, loss_rec = 0, 0, 0, 0
            l1_loss, contrast_loss = criterion.l1_loss_value, criterion.contrast_loss_value
            # l1_loss, contrast_loss = 0, 0
            
            if args.return_seg:
                sl_loss = nn.L1Loss()(pred_seg, gt_seg.detach())
                sl_weight = 0.05
                loss += sl_weight * sl_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if ((epoch + 1) % args.display_iter) == 0:
                print("At iteration %d  loss: %.8f l1_loss: %.4f contrast_loss: %.4f semantic_loss: %.4f lr: %.6f" %
                      (epoch + 1, loss.item(), l1_loss, contrast_loss, sl_loss.item(), lr))
            if ((epoch + 1) % args.snapshot_iter) == 0:
                save_model = dehaze_net.module.state_dict() if args.multi_gpu else dehaze_net.state_dict()
                torch.save(save_model, args.snapshots_folder + '/latest.pth')
                current_psnr, current_ssim = valid(args, dehaze_net, val_loader, refinenet)
                if current_psnr > best_psnr or current_ssim > best_ssim:
                    print('===> save best...')
                    best_psnr = max(best_psnr, current_psnr)
                    best_ssim = max(best_ssim, current_ssim)
                    best_epoch = epoch + 1
                    torch.save(save_model, args.snapshots_folder + "/best_dehazer.pth")
        print("Best psnr: %.4f, best ssim: %.4f at epoch %d" % (best_psnr, best_ssim, best_epoch))


def valid(args, dehaze_net, val_loader, refinenet=None):
    print('\nStart validation!')

    dehaze_net.eval()
    total_psnr = 0
    total_ssim = 0
    for iter_val, data in enumerate(val_loader):
        if args.resize:
            raise NotImplementedError('Resize as the size of training before test, and resize back after test.')
        
        if args.return_seg:
            img_hazy, img_gt, img_hazy2seg = data
            img_hazy = img_hazy.cuda()
            img_gt = img_gt.cuda()
            img_hazy2seg = img_hazy2seg.cuda()
            with torch.no_grad():
                img_seg = refinenet(img_hazy2seg)
                img_pred = dehaze_net(img_hazy, img_seg)
                img_pred = torch.clamp(img_pred, 0, 1)
        else:
            img_hazy, img_gt = data
            img_hazy = img_hazy.cuda()
            img_gt = img_gt.cuda()
            with torch.no_grad():
                img_pred = dehaze_net(img_hazy)
                img_pred = torch.clamp(img_pred, 0, 1)

        psnr, ssim = eval_index(img_gt, img_pred)
        total_psnr += psnr
        total_ssim += ssim

        if args.valid_save:
            torchvision.utils.save_image(torch.cat((img_hazy, img_pred, img_gt), 0),
                                         args.sample_output_folder + '/' + str(iter_val + 1) + ".jpg")

    total_psnr = total_psnr / args.val_total_num * args.val_batch_size
    total_ssim = total_ssim / args.val_total_num * args.val_batch_size
    print('Validate PSNR: %.4f' % total_psnr)
    print('Validate SSIM: %.4f' % total_ssim)
    return total_psnr, total_ssim

def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--data_name', type=str, default='ITS', choices=['ITS', 'OTS', 'NH-HAZE', 'DENSE-HAZE'])
    parser.add_argument('--dir_path', type=str, default="../../../Dataset/RESIDE")
    lr = parser.add_argument_group(title='Learning rate')
    lr.add_argument('--init_lr', type=float, default=0.0002)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--crop_size', type=int, default=240)
    parser.add_argument('--crop_h', type=int, default=240)
    parser.add_argument('--crop_w', type=int, default=240)  # 320
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--loss_func', default='l1', help='l1|l2')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=20)
    parser.add_argument('--snapshots_folder', type=str, default="./train_logs")
    parser.add_argument('--sample_output_folder', type=str, default="./train_logs/samples")
    parser.add_argument('--model_path', type=str, default='./train_logs/best_dehazer.pth')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--model_trained', action='store_true')
    parser.add_argument('--valid_save', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--contrast_w', type=float, default=0.1)
    parser.add_argument('--neg_num', type=int, default=0)
    
    parser.add_argument('--model_mode', type=str, default='SG', choices=['base', 'SG'])
    parser.add_argument('--seg_ckpt_path', type=str, default='../../semantic_segmentation/light-weight-refinenet/ckpt/50_nyud.ckpt')
    parser.add_argument('--seg_model_res', type=str, default='50', choices=['50', '101', '152'])
    parser.add_argument('--seg_dataset', type=str, default='nyud', choices=['nyud', 'voc'])
    parser.add_argument('--seg_class_num', type=int, default=40)  # nyud: 40, voc: 21
    parser.add_argument('--seg_dim', type=int, default=16) 
    args = parser.parse_args()
    args.return_seg = args.model_mode != 'base'
    rs = args.return_seg

    if not os.path.exists(args.snapshots_folder):
        os.system('mkdir -p %s' % args.snapshots_folder)
    if args.valid_save and not os.path.exists(args.sample_output_folder):
        os.system('mkdir -p %s' % args.sample_output_folder)

    t1 = time.time()
    # set_seed_torch(666)

    if args.data_name == 'NH-HAZE':
        args.dir_path = "../../../Dataset/NH-HAZE"
        train_dataset = NH_HAZE(args.dir_path, mode='train', crop=args.crop, crop_h=args.crop_h, crop_w=args.crop_w, return_seg=rs)
        test_dataset = NH_HAZE(args.dir_path, mode='test', return_seg=rs)
    elif args.data_name == 'DENSE-HAZE':
        args.dir_path = "../../../Dataset/DENSE-HAZE"
        train_dataset = DENSE_HAZE(args.dir_path, mode='train', crop=args.crop, crop_h=args.crop_h, crop_w=args.crop_w, return_seg=rs)
        test_dataset = DENSE_HAZE(args.dir_path, mode='test', return_seg=rs)
    elif args.data_name == 'ITS':
        args.dir_path = "../../../Dataset/RESIDE"
        train_dataset = RESIDE_Dataset(args.dir_path + '/Standard/ITS', train=True, size=args.crop_size, return_seg=rs)
        test_dataset = RESIDE_Dataset(args.dir_path + '/Standard/SOTS/indoor', train=False, size='whole img', return_seg=rs)
    elif args.data_name == 'OTS':
        args.dir_path = "../../../Dataset/RESIDE"
        train_dataset = RESIDE_Dataset(args.dir_path + '/Beta/OTS', train=True, size=args.crop_size, format='.jpg', return_seg=rs)
        test_dataset = RESIDE_Dataset(args.dir_path + '/Standard/SOTS/outdoor', train=False, size="whole img", return_seg=rs)
    
    for k, v in args.__dict__.items():
        print(k, ': ', v)
    
    print('Data loaded. Train num: %d, valid num: %d.' % (len(train_dataset), len(test_dataset)))
    args.val_total_num = len(test_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.val_batch_size, shuffle=False)
    val_loader = test_loader
    
    if args.model_mode == 'base':
        dehaze_net = Dehaze(input_nc=3, output_nc=3).cuda()
    elif args.model_mode == 'SG':
        dehaze_net = SGDehaze(input_nc=3, output_nc=3, seg_dim=args.seg_dim, num_classes=args.seg_class_num).cuda()
    else:
        raise NotImplementedError

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
 
    train(args, dehaze_net, train_loader, val_loader, refinenet)
    print('Time consume: %.2f h' % ((time.time() - t1) / 3600))
