import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.RSU7 import RSU7
from Utils.RSU6 import RSU6
from Utils.RSU5 import RSU5
from Utils.RSU4 import RSU4
from Utils.RSU4F import RSU4F

# MRNet
class MRNet(nn.Module):
    def __init__(self, in_ch=1, out_channel=2):
        super(MRNet, self).__init__()

        self.stage1 = RSU7(in_ch, 8, 8)
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(8, 8, 32)
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(32, 16, 64)
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 32, 128)
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(128, 64, 256)

        self.outpool = nn.AdaptiveMaxPool3d(1)

        self.linear = nn.Linear(256, out_channel)

    def forward(self, x):
        #stage 1
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)

        # out
        featureforLinearlayer = self.outpool(hx5)
        featureforLinearlayer = featureforLinearlayer.view(featureforLinearlayer.size(0), -1)
        Linearlayer_out = self.linear(featureforLinearlayer)
        out_label = F.softmax(Linearlayer_out, dim=1)

        return out_label