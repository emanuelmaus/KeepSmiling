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
from model import Encoder
from utils import weights_init 


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--mode', type=str, default='testing', choices=['training', 'testing'])
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

# Create directories if not exist.
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# Get parameters
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)


# Our Dataset
mtfl_loader = get_loader(opt.dataroot, int(3), int(3), opt.imageSize,
                             opt.batchSize, (opt.mode=='training'), opt.workers)

# Create the Encoder network and load its trained parameters
netE = Encoder(ngpu).to(device)
netE.apply(weights_init)
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
print(netE)

# Calculate the feature vectors

# Feature_vector
smiling_feature_vectors = []
not_smiling_feature_vectors = []
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
    
    # label_value == 0 --> smiling
    if(label_value == 0):
        feature_vector = netE(image)
        # Transform it to a 1D-Vector
        feature_vector = feature_vector.view(-1, 1*128*5*5).transpose(0, 1)
        smiling_feature_vectors.append(feature_vector)
    else:
        feature_vector = netE(image)
        # Transform it to a 1D-Vector
        feature_vector = feature_vector.view(-1, 1*128*5*5).transpose(0, 1)
        not_smiling_feature_vectors.append(feature_vector)

smile_tensor = torch.stack(smiling_feature_vectors)
not_smile_tensor = torch.stack(not_smiling_feature_vectors)

# To numpy
smile_numpy = smile_tensor.data.cpu().numpy()
print(smile_numpy.shape)
not_smile_numpy = not_smile_tensor.data.cpu().numpy()
print(not_smile_numpy.shape)

# Save the numpy array
np.save(os.path.join(opt.outf,'smile_vec.npy'), smile_numpy)        
np.save(os.path.join(opt.outf,'not_smile_vec.npy'), not_smile_numpy)        
    

