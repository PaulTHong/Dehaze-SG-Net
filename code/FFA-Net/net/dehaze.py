import os,argparse
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import glob
import time
import cv2

def dehaze_piece_image(dehaze_net, data_haze, clean_image, i1, i2, j1, j2, pad):
    haze_piece = data_haze[:, :, i1:i2 + 2 * pad, j1:j2 + 2 * pad]
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
        pad_haze = torch.zeros(1, 3, H + 2 * pad, W + 2 * pad)
        pad_haze[:, :, pad:-pad, pad:-pad] = data_haze
        data_haze = pad_haze.cuda()

    clean_image = np.zeros((3, H, W))
    h_count = H // h_piece
    w_count = W // w_piece
    h_left = H % h_piece
    w_left = W % w_piece
    for i in range(h_count):
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i * h_piece, (i + 1) * h_piece,
                                             j * w_piece, (j + 1) * w_piece, pad)
        if w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i * h_piece, (i + 1) * h_piece,
                                             j * w_piece, W + 1, pad)
    if h_left > 0:
        i = h_count
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i * h_piece, H + 1, j * w_piece,
                                             (j + 1) * w_piece, pad)
        if w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i * h_piece, H + 1, j * w_piece, W + 1,
                                             pad)
    return clean_image


def post_process(image):
    pass


def dehaze_image(image_path, net, args):
    data_haze = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    GRAY = False
    if len(data_haze.shape) == 3 and data_haze.shape[2] == 3:
        data_haze = cv2.cvtColor(data_haze, cv2.COLOR_BGR2RGB)
    elif len(data_haze.shape) == 2:
        print('not 3 channel, image shape is: ', data_haze.shape)
        data_haze = cv2.cvtColor(data_haze, cv2.COLOR_GRAY2RGB)
        GRAY = True
    else:
        print('not 3 channel, image shape is: ', data_haze.shape)
        data_haze = cv2.imread(image_path, cv2.IMREAD_COLOR)
        data_haze = cv2.cvtColor(data_haze, cv2.COLOR_BGR2RGB)
    
    assert len(data_haze.shape) == 3, 'check the channel of image is equal to 3'
    # data_haze = data_haze / 255.0
    data_haze = Image.fromarray(data_haze)
    haze1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(data_haze)[None, ::]
    haze_no = tfs.ToTensor()(data_haze)[None, ::]

    haze1 = haze1.cuda()
    haze_no = haze_no.cuda()
    dehaze_net = net

    with torch.no_grad():
        if not args.crop:
            clean_image = dehaze_net(haze1)
            clean_image = clean_image.cpu().detach().numpy()[0]
            save_path = os.path.join(args.output_dir, os.path.split(image_path)[-1][:-4] + '.png')
        else:
            clean_image = crop_splice_image(dehaze_net, haze1, args)
            save_path = os.path.join(args.output_dir,
                                     'smooth' + str(args.pad) + '_' + os.path.split(image_path)[-1][:-4] + '.png')

    clean_image = clean_image.transpose((1, 2, 0))
    show_clean_image = np.clip(clean_image * 255, 0, 255)

    if GRAY:
        show_clean_image = np.uint8(show_clean_image)
        show_clean_image = cv2.cvtColor(show_clean_image, cv2.COLOR_RGB2GRAY)
    else:
        show_clean_image = cv2.cvtColor(show_clean_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, show_clean_image)


abs=os.getcwd()+'/'
def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_imgs',help='Test imgs folder')
parser.add_argument('--output_dir', type=str, default='./results')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--h_piece', type=int, default=512)
parser.add_argument('--w_piece', type=int, default=512)
parser.add_argument('--pad', type=int, default=3)
# parser.add_argument('--post_process', action='store_true')
opt=parser.parse_args()
dataset=opt.task
gps=3
blocks=19
img_dir=abs+opt.test_imgs+'/'
output_dir=abs+f'pred_FFA_{dataset}/'
opt.output_dir = output_dir
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir=abs+f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=FFA(gps=gps,blocks=blocks)
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

test_list = glob.glob(os.path.join(img_dir, '*'))
for image in test_list:
    if os.path.isdir(image):
        continue
    t1 = time.time()
    dehaze_image(image, net, opt)
    print(os.path.split(image)[-1], "done!")
    print('Time used:%.2fs' % (time.time()-t1))


