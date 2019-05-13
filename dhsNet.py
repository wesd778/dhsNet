# coding=utf-8
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class gatelayer(nn.Module):
    def __init__(self, datain, dataout, ksize, ssize):
        super(gatelayer, self).__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(datain, dataout, (ksize, 1), (1, 1), (ssize, 0)),
            nn.BatchNorm2d(dataout),
            nn.LeakyReLU(),
            nn.Sigmoid(),
        )
        self.con = nn.Sequential(
            nn.Conv2d(datain, dataout, (ksize, 1), (1, 1), (ssize, 0)),
            nn.BatchNorm2d(dataout),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        y1 = self.gate(x)
        y2 = self.con(x)
        return y1 * y2


class Incep1(nn.Module):
    def __init__(self):
        super(Incep1, self).__init__()

        self.x1 = gatelayer(4, 96, 1, 0)
        self.x2 = gatelayer(4, 96, 3, 1)
        self.x3 = gatelayer(4, 96, 5, 2)
        self.x4 = gatelayer(4, 96, 7, 3)

    def forward(self, x):
        y1 = self.x1(x)
        y2 = self.x2(x)
        y3 = self.x3(x)
        y4 = self.x4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class Incep2(nn.Module):
    def __init__(self):
        super(Incep2, self).__init__()

        self.mymode1 = nn.Sequential(
            nn.AvgPool2d((3, 1), (1, 1), (1, 0)),
            gatelayer(384, 96, 1, 0),
        )
        self.mymode2 = gatelayer(384, 96, 1, 0)
        self.mymode3 = nn.Sequential(
            gatelayer(384, 64, 1, 0),
            gatelayer(64, 96, 7, 3),
        )
        self.mymode4 = nn.Sequential(
            gatelayer(384, 64, 1, 0),
            gatelayer(64, 96, 7, 3),
        )

    def forward(self, x):
        y1 = self.mymode1(x)
        y2 = self.mymode2(x)
        y3 = self.mymode3(x)
        y4 = self.mymode4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class Incep3(nn.Module):
    def __init__(self):
        super(Incep3, self).__init__()

        self.mymode1 = nn.Sequential(
            nn.AvgPool2d((3, 1), (1, 1), (1, 0)),
            gatelayer(384, 96, 1, 0),
        )
        self.mymode2 = gatelayer(384, 96, 1, 0)
        self.mymode3 = nn.Sequential(
            gatelayer(384, 64, 1, 0),
            gatelayer(64, 96, 11, 5),
        )
        self.mymode4 = nn.Sequential(
            gatelayer(384, 64, 1, 0),
            gatelayer(64, 96, 11, 5),
        )

    def forward(self, x):
        y1 = self.mymode1(x)
        y2 = self.mymode2(x)
        y3 = self.mymode3(x)
        y4 = self.mymode4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class SPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='avg_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, w, h, c = x.size()
        pooling_layers = []

        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=(kernel_size, 1), stride=kernel_size)
            else:
                tensor = F.avg_pool2d(x, kernel_size=(kernel_size, 1), stride=kernel_size)

            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, 2)

        return x


class MyNet(nn.Module):
    def __init__(self, spp_level=3, num_grids=7):
        super(MyNet, self).__init__()

        self.a0 = Incep1()
        self.p0 = nn.MaxPool2d((2, 1))
        self.d0 = nn.Dropout(0.3)

        self.a1 = Incep2()
        self.p1 = nn.MaxPool2d((2, 1))
        self.d1 = nn.Dropout(0.3)

        self.a2 = Incep2()
        self.p2 = nn.MaxPool2d((2, 1))
        self.d2 = nn.Dropout(0.3)

        self.a3 = Incep3()
        self.p3 = nn.MaxPool2d((2, 1))
        self.d3 = nn.Dropout(0.3)

        self.a4 = Incep3()
        self.p4 = SPPLayer(spp_level)
        self.d4 = nn.Dropout(0.3)

        self.r2 = nn.Linear(num_grids * 384, 384)
        self.r3 = nn.BatchNorm1d(384)
        self.r4 = nn.LeakyReLU()
        self.r5 = nn.Dropout(0.3)

        self.r6 = nn.Linear(384, 384)
        self.r7 = nn.BatchNorm1d(384)
        self.r8 = nn.LeakyReLU()
        self.r9 = nn.Dropout(0.3)

        self.r10 = nn.Linear(384, 2)
        self.r11 = nn.BatchNorm1d(2)
        self.r12 = nn.LeakyReLU()
        self.r13 = nn.Dropout(0.3)

    def forward(self, x, num_grids=7):
        out = self.a0(x)
        out = self.p0(out)
        out = self.d0(out)

        out = self.a1(out)
        out = self.p1(out)
        out = self.d1(out)

        out = self.a2(out)
        out = self.p2(out)
        out = self.d2(out)

        out = self.a3(out)
        out = self.p3(out)
        out = self.d3(out)
        out = self.a4(out)
        out = self.p4(out)
        out = self.d4(out)

        if out.size()[2] != 7:
            out = out[:, :, :7, :].contiguous()

        out = out.view(-1, num_grids * 384)
        # out = out.view(-1, 384)

        out = self.r2(out)
        out = self.r3(out)
        out = self.r4(out)
        out = self.r5(out)
        out1 = out

        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out2 = out

        out = self.r10(out)
        out = self.r11(out)
        out = self.r12(out)
        out = self.r13(out)
        out3 = out

        return out


net = MyNet()
