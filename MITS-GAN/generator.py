import torch
import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, in_channels, filters=64, ngpu=1, kernel_size=3, stride=1, padding=1):
        super(residual_block, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters)
            )

    def forward(self, inputs):
        output = self.main(inputs)
        output += self.shortcut(inputs)
        return output

class Noisenet(nn.Module):
    def __init__(self):
        super(Noisenet, self).__init__()
        self.net_noise = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2, 4, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 4, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2, 1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

    def forward(self, input):
        noise_protection = self.net_noise(input)
        return noise_protection

class Generator(nn.Module):
    def __init__(self, ngpu, out_channels, image_size, kernel_size=3, stride=1, padding=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        filters = 64
        blocks = []
        for block in range(3):
            blocks.append(residual_block(filters, filters, ngpu, kernel_size, stride, padding))
        
        self.noisenet = Noisenet()

        self.main = nn.Sequential(
            nn.Conv2d(2, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            *blocks,
            nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        noise_protection = self.noisenet(torch.rand((input.size(0),1,512,512)).double().cuda())
        input = torch.cat((noise_protection, input), 1)
        output = self.main(input)
        return output
