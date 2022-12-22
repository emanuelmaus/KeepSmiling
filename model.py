""" Containing the used models  """

import torch.nn as nn

# Generator network classes (rand-vector --> image)

## Create a Residual Block for the Generator (splitGAN and complexGAN
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

## Sinple Generator network
class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nz, nc):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.ngf = ngf
        self.nz = nz
        self.nc = nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2,     self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    self.ngf,      self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

## More comeplex structured Generator network (complexGAN setup)
class ComplexGenerator(nn.Module):
    def __init__(self, ngpu, ngf, nc, conv_dim=32, class_dim=1, repeat_num=2):
        super(ComplexGenerator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.nc = nc
        self.conv_dim = conv_dim
        self.class_dim = class_dim
        self.repeat_num = repeat_num

        #Layers, which convert the image into the feature space
        layers_to_feature_space=[]
        layers_to_feature_space.append(nn.Conv2d(self.nc+self.class_dim, self.conv_dim, kernel_size=7, stride=3,
                                        padding=3, bias=False))
        layers_to_feature_space.append(nn.InstanceNorm2d(self.conv_dim, affine=True,
                                        track_running_stats=True))
        layers_to_feature_space.append(nn.ReLU(inplace=True))

        #Downsampling layers
        curr_dim = self.conv_dim
        for i in range(2):
            layers_to_feature_space.append(nn.Conv2d(curr_dim, curr_dim*2,
                                                     kernel_size=4, stride=2, padding=1, bias=False))
            layers_to_feature_space.append(nn.InstanceNorm2d(curr_dim*2, affine=True,
                                                             track_running_stats=True))
            layers_to_feature_space.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        #Bottleneck layers
        for i in range(self.repeat_num):
            layers_to_feature_space.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        #Layers, which convert feature vector into a image

        layers_to_image_space = []
        # input is feature vector, going into a convolution
        layers_to_image_space.append(nn.ConvTranspose2d(curr_dim, self.ngf * 8, 4, 1, 0, bias=False))
        layers_to_image_space.append(nn.BatchNorm2d(self.ngf * 8))
        layers_to_image_space.append(nn.ReLU(True))
        # state size. (ngf*8) x 8 x 8
        layers_to_image_space.append(nn.ConvTranspose2d(self.ngf * 8, ngf * 4, 4, 2, 1, bias=False))
        layers_to_image_space.append(nn.BatchNorm2d(self.ngf * 4))
        layers_to_image_space.append(nn.ReLU(True))
        # state size. (ngf*4) x 16 x 16
        layers_to_image_space.append(nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False))
        layers_to_image_space.append(nn.BatchNorm2d(self.ngf * 2))
        layers_to_image_space.append(nn.ReLU(True))
        # state size. (ngf*2) x 32 x 32
        layers_to_image_space.append(nn.ConvTranspose2d(self.ngf * 2, self.nc, 4, 2, 1, bias=False))
        layers_to_image_space.append(nn.Tanh())
        # state size. (nc) x 64 x 64

        #Combine both layers
        all_layers = layers_to_feature_space + layers_to_image_space


        self.main = nn.Sequential(*all_layers)


    def forward(self, input, cls):

        # Replicate spatially and concatenate domain information.
        cls = cls.view(cls.size(0), 1, 1, 1)    #modified second entry from cls.size(1) to 1
        cls = cls.repeat(1, 1, input.size(2), input.size(3))
        input = torch.cat([input, cls], dim=1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

## Encoder and Decoder model for the splitGAN experiment
### Encoder
class Encoder(nn.Module):
    def __init__(self, ngpu, nc, conv_dim=32, repeat_num=2):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.conv_dim = conv_dim
        self.repeat_num = repeat_num

        #Layers, which convert the image into the feature space
        layers_to_feature_space = []
        layers_to_feature_space.append(nn.Conv2d(self.nc, self.conv_dim, kernel_size=7, stride=3,
                                        padding=3, bias=False))
        layers_to_feature_space.append(nn.InstanceNorm2d(self.conv_dim, affine=True,
                                        track_running_stats=True))
        layers_to_feature_space.append(nn.ReLU(inplace=True))

        #Downsampling layers
        curr_dim = self.conv_dim
        for _ in range(2):
            layers_to_feature_space.append(nn.Conv2d(curr_dim, curr_dim*2,
                                                     kernel_size=4, stride=2, padding=1, bias=False))
            layers_to_feature_space.append(nn.InstanceNorm2d(curr_dim*2, affine=True,
                                                             track_running_stats=True))
            layers_to_feature_space.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        #Bottleneck layers
        for _ in range(self.repeat_num):
            layers_to_feature_space.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))


        self.main = nn.Sequential(*layers_to_feature_space)


    def forward(self, input):

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

### Decoder
class Decoder(nn.Module):
    def __init__(self, ngpu, nc,  ngf,  curr_dim=128, conv_dim=32, repeat_num=2):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ngf = ngf
        self.curr_dim = curr_dim
        self.conv_dim = conv_dim
        self.repeat_num = repeat_num

        #Layers, which convert feature vector into a image

        layers_to_image_space = []
        # input is feature vector, going into a convolution
        layers_to_image_space.append(nn.ConvTranspose2d(curr_dim, self.ngf * 8, 4, 1, 0, bias=False))
        layers_to_image_space.append(nn.BatchNorm2d(ngf * 8))
        layers_to_image_space.append(nn.ReLU(True))
        # state size. (ngf*8) x 8 x 8
        layers_to_image_space.append(nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False))
        layers_to_image_space.append(nn.BatchNorm2d(self.ngf * 4))
        layers_to_image_space.append(nn.ReLU(True))
        # state size. (ngf*4) x 16 x 16
        layers_to_image_space.append(nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False))
        layers_to_image_space.append(nn.BatchNorm2d(self.ngf * 2))
        layers_to_image_space.append(nn.ReLU(True))
        # state size. (ngf*2) x 32 x 32
        layers_to_image_space.append(nn.ConvTranspose2d(self.ngf * 2, self.nc, 4, 2, 1, bias=False))
        layers_to_image_space.append(nn.Tanh())
        # state size. (nc) x 64 x 64


        self.main = nn.Sequential(*layers_to_image_space)


    def forward(self, input):

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# Discriminator (discriminates if fake or real image)

## Sinple Discriminator (simpleGAN and splitGAN setup)
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


## ComplexDiscriminator for the complexGAN setup
class ComplexDiscriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(ComplexDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf

        self.smile_NN = []
        self.fake_NN = []
        self.main = []

        #Create main NN
        # input is (nc) x 64 x 64
        self.main.append(nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False))
        self.main.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 32 x 32
        self.main.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.main.append(nn.BatchNorm2d(self.ndf * 2))
        self.main.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*2) x 16 x 16
        self.main.append(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.main.append(nn.BatchNorm2d(self.ndf * 4))
        self.main.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*4) x 8 x 8
        self.main.append(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False))
        self.main.append(nn.BatchNorm2d(self.ndf * 8))
        self.main.append(nn.LeakyReLU(0.2, inplace=True))

        #Create Fake-NN
        # state size. (ndf*8) x 4 x 4
        fake_temp = []
        fake_temp.append(nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False))
        fake_temp.append(nn.Sigmoid())

        self.fake_NN = self.main + fake_temp

        #Create Smile-NN
        smile_temp = []
        smile_temp.append(nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False))
        smile_temp.append(nn.Sigmoid())

        self.smile_NN = self.main + smile_temp

        #Create the sequentials
        self.sequential_fake = nn.Sequential(*self.fake_NN)
        self.sequential_smile = nn.Sequential(*self.smile_NN)


    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output_src = nn.parallel.data_parallel(self.sequential_fake, input, range(self.ngpu))
            output_cls = nn.parallel.data_parallel(self.sequential_smile, input, range(self.ngpu))
        else:
            output_src = self.sequential_fake(input)
            output_cls = self.sequential_smile(input)
        return output_src.view(-1, 1).squeeze(1), output_cls.view(-1, 1).squeeze(1)
