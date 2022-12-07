from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils

from data_loader import get_loader 
from model import Encoder, Decoder
from utils import weights_init, gradient_penalty 


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--mode', type=str, default='testing', choices=['training', 'testing'])
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netDe', default='', help="path to netDe (to continue training)")
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

# Create directories if not exist.
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Our Dataset
mtfl_loader = get_loader(opt.dataroot, int(3), int(3), opt.imageSize,
                             opt.batchSize, (opt.mode=='training'), opt.workers)

# Parameters
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
nc = 3

# Encoder
netE = Encoder(ngpu, nc).to(device)
netE.apply(weights_init)
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
print(netE)


# Decoder
netDe = Decoder(ngpu, nc, ngf).to(device)
netDe.apply(weights_init)
if opt.netDe != '':
    netDe.load_state_dict(torch.load(opt.netDe))
print(netDe)


# Load the translation vector
translation_vec = np.load(os.path.join(opt.outf,'translation_to_smile_vec.npy'))
translation_vec = torch.from_numpy(translation_vec).to(device)

# Transformation of the data (Translation in feature space)
print(len(mtfl_loader))
for i, data in enumerate(mtfl_loader, 0):
    image = data[0].to(device)
    label = data[1]
    label_value = int(label.view(-1, 1)[0])
    
    if(image.size() != (1,3,64,64)):
        print(i)
        print("Not the correct size")
        print(image.size())
        continue
    
    #label_value == 0 --> smiling
    if(label_value == 0):
        feature_vector = netE(image)
        #Transform it to a 1D-Vector and translate it
        feature_vector = feature_vector.view(-1, 1*128*5*5).transpose(0, 1) - translation_vec
        not_smiling = netDe(feature_vector.view(1,128,5,5))
        save_list = [image, not_smiling]
        save_cat = torch.cat(save_list, dim=(len(not_smiling.size())-1))
        save_path = os.path.join(opt.outf, '{}_not_smiling_images.png'.format(i+1))
        vutils.save_image(((save_cat.data.cpu() + 1)/2).clamp_(0,1), save_path, nrow=1, padding=0)
        
    else:
        feature_vector = netE(image)
        #Transform it to a 1D-Vector
        feature_vector = feature_vector.view(-1, 1*128*5*5).transpose(0, 1) + translation_vec
        smiling = netDe(feature_vector.view(1,128,5,5))
        save_list = [image, smiling]
        save_cat = torch.cat(save_list, dim=(len(smiling.size())-1))
        save_path = os.path.join(opt.outf, '{}_smiling_images.png'.format(i+1))
        vutils.save_image(((save_cat.data.cpu() + 1)/2).clamp_(0,1), save_path, nrow=1, padding=0)
    
    

