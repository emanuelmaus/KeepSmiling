from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from data_loader import get_loader
from logger import Logger 
from utils import weights_init, gradient_penalty
from model import Encoder, Decoder, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'])
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nstart', type=int, default=0, help='start number of epochs to train for')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netDe', default='', help="path to netDe (to continue training)")
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Create directories if not exist.
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

subfolder1=os.path.join(opt.outf,'netE_folder')
subfolder2=os.path.join(opt.outf,'netDe_folder')
subfolder3=os.path.join(opt.outf,'netD_folder')
subfolder4=os.path.join(opt.outf,'log_folder')

if not os.path.exists(subfolder1):
    os.makedirs(subfolder1)
if not os.path.exists(subfolder2):
    os.makedirs(subfolder2)
if not os.path.exists(subfolder3):
    os.makedirs(subfolder3)
if not os.path.exists(subfolder4):
    os.makedirs(subfolder4)

# Parameters
cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

# Our Dataset
mtfl_loader = get_loader(opt.dataroot, int(3), int(1), opt.imageSize,
                             opt.batchSize, (opt.mode=='training'), opt.workers)

# Build tensorboard
logger = Logger(subfolder4)

## Create Generator network
# Create and initialise/load Encoder network
netE = Encoder(ngpu, nc).to(device)
netE.apply(weights_init)
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
print(netE)

# Create and initialise/load Decoder network
netDe = Decoder(ngpu, nc, ngf).to(device)
netDe.apply(weights_init)
if opt.netDe != '':
    netDe.load_state_dict(torch.load(opt.netDe))
print(netDe)


## Create Discriminator network
# Create and initialise/load Decoder network
netD = Discriminator(ngpu, nc, ndf).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Loss criterion
criterion = nn.BCELoss()

# Fetch fixed inputs for debugging.
data_iter = iter(mtfl_loader)
x_fixed,_ = next(data_iter)
x_fixed = x_fixed.to(device)

# Labels for real and fake
real_label = 1
fake_label = 0

# Define the learning rate
lr_D = opt.lr
lr_E = opt.lr
lr_De = opt.lr

# Setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr= lr_D, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=lr_E, betas=(opt.beta1, 0.999))
optimizerDe = optim.Adam(netDe.parameters(), lr=lr_De, betas=(opt.beta1, 0.999))

# Start time
start_time = time.time()

for epoch in range(opt.nstart, opt.niter):
    for i, data in enumerate(mtfl_loader, 0):
        ############################
        # First: update D network
        
        # Train with real
        netD.zero_grad()
        x_real = data[0].to(device)
        
        batch_size = x_real.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        
        # Compute loss with real images
        output_x_real = netD(x_real)
        
        errD_real = criterion(output_x_real, label)
        
        D_x_real = output_x_real.mean().item()
        
        # Compute feature_vector and its generated image
        feature_vector = netE(x_real)
        x_fake = netDe(feature_vector)
        
        # Train with fake images
        label_fake = label.clone().fill_(fake_label)
        output_x_fake = netD(x_fake.detach())
        D_x_fake = output_x_fake.mean().item()

        #Compute loss with fake images
        errD_fake = criterion(output_x_fake, label_fake)
        
        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        output_x_grad = netD(x_hat)
        D_x_grad = output_x_grad.mean().item()
        errD_gp = gradient_penalty(output_x_grad, x_hat, device)
        
        # Backward and optimize.
        errD_all = errD_real + errD_fake + 3*errD_gp

        # Reset the gradient buffers 
        optimizerD.zero_grad()
        optimizerE.zero_grad()
        optimizerDe.zero_grad()
        
        errD_all.backward()
        optimizerD.step()
        
        ############################

        ############################
        #Second: update G network
        
        # Train the Generator (1xtraining Generator, 2xtraining Discriminator)
        if((i+1) % 2 == 0 or (i==0)):
            # Original-to-rand domain
            feature_vector = netE(x_real)
            x_fake = netDe(feature_vector)
                  		
            netE.zero_grad()
            netDe.zero_grad()
            
            # Discriminate the fake image
            label_real = label.clone().fill_(real_label)  # fake labels are real for generator cost
            output_x3 = netD(x_fake)
            D_G_x3 = output_x3.mean().item()
            errG_fake = criterion(output_x3, label_real)
            
            # Target-to-original domain.
            feature_vector = netE(x_fake)
            x_reconst = netDe(feature_vector)
            D_G_x_rec = x_reconst.mean().item()
            errG_rec = torch.mean(torch.abs(x_real - x_reconst))

            
            # Backward and optimize (weights determined empirically)
            errG_all = 1*errG_fake + 9*errG_rec
            
            
            errG_all.backward()
            optimizerDe.step()
            optimizerE.step()
            
	    # Reset the gradient buffers 
            optimizerDe.zero_grad()
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            ############################   
                  
    
        # Get Output(states and images)

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z(x))): %.4f / %.4f D(x,G(x)) %.4f'
              % (epoch, opt.niter, i, len(mtfl_loader),
                 errD_all.item(), errG_all.item(), D_x_real, D_x_fake, D_G_x3, D_x_grad))
        if i % 100 == 0:
            vutils.save_image(x_fixed,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            feature_vector = netE(x_fixed)
            fake = netDe(feature_vector)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
    
    # Logger data per epoch
    # Logging
    # Diskriminator
    loss = {}
    loss['D/loss_real'] = D_x_real
    loss['D/loss_fake'] = D_x_fake
    loss['D/loss_x_grad'] = D_x_grad    

    # Generator
    loss['G/loss_fake'] = D_G_x3
    loss['G/loss_rec'] = D_G_x_rec    
    
    delta_t = time.time() - start_time
    delta_t = str(delta_t)[:-5]
    log = "Elapsed [{}], Iteration [{}/{}]".format(delta_t, epoch+1,  opt.niter)
    
    # Print the log
    for tag, value in loss.items():
        log += ", {}: {:.4f}".format(tag, value)
        logger.scalar_summary(tag, value, epoch+1)
    print("New epoche: log")    
    print(log)      
    print("##########")

    # Decay learning rates.
    if (epoch+1) % 10 == 0 and (epoch+1) > (opt.niter - 70):
        lr_D -= (lr_D / float(70))
        lr_E -= (lr_E / float(70))
        lr_De -= (lr_De / float(70))
        # Update the learning rates
        for param_group in optimizerD.param_groups:
            param_group['lr'] = lr_D
        for param_group in optimizerE.param_groups:
            param_group['lr'] = lr_E
        for param_group in optimizerDe.param_groups:
            param_group['lr'] = lr_De
        
        print("Decay learning rate:")
        print('Decayed learning rates, lr_D: {}, lr_E: {}, lr_De: {}.'.format(lr_D, lr_E, lr_De))
        print("####################")

    # Do checkpointing
    torch.save(netDe.state_dict(), '%s/netDe_folder/netDe_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netE.state_dict(), '%s/netE_folder/netE_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_folder/netD_epoch_%d.pth' % (opt.outf, epoch))
