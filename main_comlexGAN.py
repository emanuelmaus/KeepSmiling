from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils


from data_loader import get_loader
from utils import weights_init, gradient_penalty, StableBCELoss
from model import ComplexGenerator, ComplexDiscriminator 


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'])
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nstart', type=int, default=0, help='startnumber of epochs to train for')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

# Create directories if not exist.
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

subfolder1=os.path.join(opt.outf,'netG_folder')
subfolder2=os.path.join(opt.outf,'netD_folder')

if not os.path.exists(subfolder1):
    os.makedirs(subfolder1)
if not os.path.exists(subfolder2):
    os.makedirs(subfolder2)


cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Our Dataset
mtfl_loader = get_loader(opt.dataroot, int(3), int(1), opt.imageSize,
                             opt.batchSize, (opt.mode=='training'), opt.workers)


device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

# Create ComplexGenerator and initialise/load the weights
netG = ComplexGenerator(ngpu, ngf, nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Create ComplexDiscriminator and initialise/load the weigths
netD = ComplexDiscriminator(ngpu, nc, ndf).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Didnt work, because of instability
#criterion = nn.BCEWithLogitsLoss()

# Therefore we used the StableBCELoss (see utils)
criterion = StableBCELoss()

# Fetch fixed inputs for debugging.
data_iter = iter(mtfl_loader)
x_fixed, labels_fixed = next(data_iter)
x_fixed = x_fixed.to(device)
labels_fixed = labels_fixed.to(device).view(-1, 1).squeeze(1)


# Setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.nstart, opt.niter):
    for i, data in enumerate(mtfl_loader, 0):
        ############################
        # First: update D network

        # Train with real
        netD.zero_grad()
        x_real = data[0].to(device)
        # Modified-Smile
        label_smile = data[1]
        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_smile.size(0))
        label_rand = label_smile[rand_idx]
        # Paste it to cuda
        label_smile = label_smile.to(device).view(-1, 1).squeeze(1)
        label_rand = label_rand.to(device).view(-1, 1).squeeze(1)
        
        # Compute loss with real images
        output_x1, output_cls1 = netD(x_real)
        
        d_loss_real = - torch.mean(output_x1)

        d_loss_cls = criterion(output_cls1, label_smile)
        
        # Compute loss with fake images
        x_fake = netG(x_real, label_rand)
        output_x2, output_cls2 = netD(x_fake.detach())
        d_loss_fake = torch.mean(output_x2)
        
        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        output_x_temp, _ = netD(x_hat)
        d_loss_gp = gradient_penalty(output_x_temp, x_hat, device)
        
        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + d_loss_cls + 10 * d_loss_gp
        # Reset the gradient buffers 
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        
        d_loss.backward()
        optimizerD.step()
        ############################
        
        ############################
        # Second: update G network
        
        # Train the Generator (1xtraining Generator, 5xtraining Discriminator)
        if((i+1) % 5 == 0 or (i==0)):
            # Original-to-rand domain
            # Train with rand-labels
            x_fake = netG(x_real, label_rand)
            output_x3, output_cls3 = netD(x_fake)
            g_loss_fake = - torch.mean(output_x3)
            g_loss_cls = criterion(output_cls3, label_rand)
            
            # Target-to-original domain.
            x_reconst = netG(x_fake, label_smile)
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
            
            # Backward and optimize.
            g_loss = g_loss_fake + g_loss_cls + 10 * g_loss_rec 
            # Reset the gradient buffers 
            optimizerG.zero_grad()
            optimizerD.zero_grad()
        
            g_loss.backward()
            optimizerG.step()

        ############################
        # Get Output(states and images)

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(mtfl_loader),
                 d_loss.item(), g_loss.item(), d_loss_real, d_loss_fake, g_loss_fake))
        if i % 100 == 0:
            vutils.save_image(x_fixed,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(x_fixed, labels_fixed)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # Do checkpointing
    torch.save(netG.state_dict(), '%s/netG_folder/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_folder/netD_epoch_%d.pth' % (opt.outf, epoch))
