from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

# Use the own dataloader
from data_loader import get_loader
from utils import weights_init  
from model import Generator, Discriminator


# Argumentparser
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'])
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

# Create directories if not exist
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

subfolder1=os.path.join(opt.outf,'netG_folder')
subfolder2=os.path.join(opt.outf,'netD_folder')

if not os.path.exists(subfolder1):
    os.makedirs(subfolder1)
if not os.path.exists(subfolder2):
    os.makedirs(subfolder2)


# Seeding of the rand-generator
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Cuda
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Our Dataset
mtfl_loader = get_loader(opt.dataroot, int(3), int(1), opt.imageSize,
                             opt.batchSize, (opt.mode=='training'), opt.workers)

# Needed initialisation
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

## Create Generator and initialise/load weights 
netG = Generator(ngpu, ngf, nz, nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

## Create Discriminator and initialise/load weights 
netD = Discriminator(ngpu, nc, ngf).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Criterion (BCELoss)
criterion = nn.BCELoss()

# Generate the noise vector (siye 100x1x1)
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

# To generate the labels/fill them
real_label = 1
fake_label = 0

# Setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(mtfl_loader, 0):
        ############################
        # First: update D network, therefore maximize log(D(x)) + log(1 - D(G(z)))
        
        # Train with real
        netD.zero_grad()
        real_image = data[0].to(device)
        batch_size = real_image.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_image)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        ###########################

        ############################
        # Second: update G network, therefore maximize log(D(G(z)))
        
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        ###########################        

        # Print the information and save real samples and generated images to compair each other
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(mtfl_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_image,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # Do checkpointing
    torch.save(netG.state_dict(), '%s/netG_folder/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_folder/netD_epoch_%d.pth' % (opt.outf, epoch))
