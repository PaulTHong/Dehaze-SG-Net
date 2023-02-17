import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16

from resnet import new_rf_lw50, new_rf_lw101, new_rf_lw152
from utils import tensor_normalize

if opt.insert_seg:
	models_ = {
		'ffa': seg_att_FFA(gps=opt.gps, blocks=opt.blocks, num_classes=opt.seg_class_num),
	}
	loaders_ = {
		'its_train': ITS_seg_train_loader,
		'its_test': ITS_seg_test_loader,
		'ots_train': OTS_seg_train_loader,
		'ots_test': OTS_seg_test_loader
	}
else:
	models_ = {
		'ffa': FFA(gps=opt.gps,blocks=opt.blocks),
	}
	loaders_ = {
		'its_train': ITS_train_loader,
		'its_test': ITS_test_loader,
		'ots_train': OTS_train_loader,
		'ots_test': OTS_test_loader
	}

mean = [0.64, 0.6, 0.58]
std = [0.14, 0.15, 0.152]
seg_mean = [0.485, 0.456, 0.406]
seg_std = [0.229, 0.224, 0.225]

refinenet = eval('new_rf_lw' + opt.seg_model_res)(num_classes=opt.seg_class_num, pretrain=True,
												  ckpt_path=opt.seg_ckpt_path).cuda()
for p in refinenet.parameters():
	p.requires_grad = False

start_time = time.time()
T = opt.steps

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
	lr = 0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net, refinenet, loader_train, loader_test, optim, criterion):
	losses = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	ssims = []
	psnrs = []
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp = torch.load(opt.model_dir)
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step = ckp['step']
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		psnrs = ['psnrs']
		ssims = ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else:
		print('train from scratch *** ')
	for step in range(start_step+1, opt.steps+1):
		net.train()
		refinenet.eval()
		lr = opt.lr
		if not opt.no_lr_sche:
			lr = lr_schedule_cosdecay(step, T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr
		x, y = next(iter(loader_train))
		x = x.to(opt.device)
		y = y.to(opt.device)
		x1 = tfs.Normalize(mean=mean, std=std)(x)
		x2 = tfs.Normalize(mean=seg_mean, std=seg_std)(x)
		img_gt2seg = tfs.Normalize(mean=seg_mean, std=seg_std)(y)
		img_seg = refinenet(x2)
		gt_seg = refinenet(img_gt2seg)

		out = net(x1, img_seg)
		img_pred2seg = tfs.Normalize(mean=seg_mean, std=seg_std)(out)
		pred_seg = refinenet(img_pred2seg)

		reconstruct_loss = criterion[0](out, y)

		if opt.perloss:
			loss2 = criterion[1](out, y)
			loss = reconstruct_loss+0.04*loss2
		sl_weight = 0.05
		sl_loss = sl_weight * nn.L1Loss()(pred_seg, gt_seg.detach())
		loss = reconstruct_loss + sl_loss

		loss.backward()

		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		print(f'\rtrain loss : {loss.item():.5f}| recon_loss: {reconstruct_loss.item():.4f}| sl_loss:{sl_loss.item():.4f}|step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}', end='', flush=True)

		if step % opt.eval_step ==0:
			with torch.no_grad():
				ssim_eval, psnr_eval = test(net, refinenet, loader_test, max_psnr, max_ssim, step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if ssim_eval > max_ssim and psnr_eval > max_psnr:
				max_ssim=max(max_ssim, ssim_eval)
				max_psnr=max(max_psnr, psnr_eval)
				torch.save({
							'step': step,
							'max_psnr': max_psnr,
							'max_ssim': max_ssim,
							'ssims': ssims,
							'psnrs': psnrs,
							'losses': losses,
							'model': net.state_dict()
				}, opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

	np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


def test(net, refinenet, loader_test, max_psnr, max_ssim,step):
	net.eval()
	refinenet.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []
	#s=True
	for i, (inputs, targets) in enumerate(loader_test):
		inputs = inputs.to(opt.device)
		targets = targets.to(opt.device)
		x1 = tfs.Normalize(mean=mean, std=std)(inputs)
		x2 = tfs.Normalize(mean=seg_mean, std=seg_std)(inputs)
		with torch.no_grad():
			img_seg = refinenet(x2)
			pred = net(x1, img_seg)

		ssim1 = ssim(pred, targets).item()
		psnr1 = psnr(pred, targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
	loader_train = loaders_[opt.trainset]
	loader_test = loaders_[opt.testset]
	net = models_[opt.net]
	net = net.to(opt.device)
	refinenet = refinenet.to(opt.device)
	if opt.device == 'cuda':
		net = torch.nn.DataParallel(net)
		refinenet = torch.nn.DataParallel(refinenet)
		cudnn.benchmark = True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
		vgg_model = vgg16(pretrained=True).features[:16]
		vgg_model = vgg_model.to(opt.device)
		for param in vgg_model.parameters():
			param.requires_grad = False
		criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()

	t1 = time.time()
	train(net, refinenet, loader_train, loader_test, optimizer, criterion)
	print('Time consume: %.2f h' % ((time.time() - t1) / 3600))


