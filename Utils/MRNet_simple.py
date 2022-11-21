import torch
import torch.nn as nn
from Utils.REBNCONV import REBNCONV
import torch.nn.functional as F

class MRNet(nn.Module):
    def __init__(self, in_ch=1, out_channel=3):
        super(MRNet, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, 16, dirate=1)

        self.rebnconv1 = REBNCONV(16, 32, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(32, 64, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(64, 128, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(128, 256, dirate=1)

        self.rebnconv5 = REBNCONV(256, 512, dirate=1)
        self.outpool = nn.AdaptiveMaxPool3d(1)
        self.linear = nn.Linear(512, out_channel)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        # out
        featureforLinearlayer = self.outpool(hx5)
        featureforLinearlayer = featureforLinearlayer.view(featureforLinearlayer.size(0), -1)
        Linearlayer_out = self.linear(featureforLinearlayer)
        out_label = F.softmax(Linearlayer_out, dim=1)

        return out_label