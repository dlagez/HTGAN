import torch.nn as nn
from DropBlock_attention import DropBlock2D
from utils import weights_init

class netG(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(netG, self).__init__()
        self.ReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)
        #self.Drop2 = nn.Dropout2d(p=0.5)
        self.Drop2 = DropBlock2D()

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ngf)

        self.conv6 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)
        x = self.Drop2(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)
        x = self.conv6(x)
        output = self.Tanh(x)
        return output

class netG_16(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(netG_16, self).__init__()
        self.ReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 2)
        #self.Drop2 = nn.Dropout2d(p=0.5)
        self.Drop2 = DropBlock2D()

        self.conv3 = nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False)
        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)
        x = self.Drop2(x)

        x = self.conv3(x)
        output = self.Tanh(x)
        return output

class netD(nn.Module):
    # 需要三个变量来控制网络的结构，
    #   ndf：终止中间层的变量
    #   nc：input channel
    #   nb_label：output class
    def __init__(self, ndf, nc, nb_label):
        super(netD, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        #self.Drop2 = nn.Dropout2d(p=0.5)
        self.Drop2 = DropBlock2D()
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False)
        #self.disc_linear = nn.Linear(ndf * 2, 1)
        self.aux_linear = nn.Linear(ndf * 2, nb_label+1)
        self.softmax = nn.LogSoftmax(dim=-1)
        #self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)
        x = self.Drop2(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)
        x = self.Drop2(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 2)
        c = self.aux_linear(x)
        c = self.softmax(c)
        #s = self.disc_linear(x).squeeze()
        #s = self.sigmoid(s)
        return c