import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size,kernel_size=4, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size=4):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=4):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, kernel_size=kernel_size, normalize=False)
        self.down2 = UNetDown(64, 128, kernel_size=kernel_size)
        self.down3 = UNetDown(128, 256, kernel_size=kernel_size)
        self.down4 = UNetDown(256, 512,kernel_size=kernel_size, dropout=0.5)
        self.down5 = UNetDown(512, 512,kernel_size=kernel_size, dropout=0.5)
        self.down6 = UNetDown(512, 512,kernel_size=kernel_size, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512,kernel_size=kernel_size, dropout=0.5)
        self.up2 = UNetUp(1024, 512,kernel_size=kernel_size, dropout=0.5)
        self.up3 = UNetUp(1024, 256,kernel_size=kernel_size, dropout=0.5)
        self.up4 = UNetUp(512, 128, kernel_size=kernel_size)
        self.up5 = UNetUp(256, 64, kernel_size=kernel_size)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, kernel_size=4):
        super(Discriminator, self).__init__()

        # Creates and returns downsampling layers of each discriminator block
        def downSamplingLayers(input_size, output_size,kernel_size=kernel_size, normalization=True):
            layers = [nn.Conv2d(input_size, output_size, kernel_size, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(output_size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *downSamplingLayers(in_channels * 2, 64, normalization=False),
            *downSamplingLayers(64, 128),
            *downSamplingLayers(128, 256),
            *downSamplingLayers(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size, padding=1, bias=False)
        )

    def forward(self, camera, scan):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((camera, scan), 1)
        return self.model(img_input)
