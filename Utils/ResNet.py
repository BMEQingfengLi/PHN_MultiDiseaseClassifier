import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.RSU7 import RSU7
from Utils.RSU6 import RSU6
from Utils.RSU5 import RSU5
from Utils.RSU4 import RSU4
from Utils.RSU4F import RSU4F

# MRNet
class ResNet(nn.Module):
    def __init__(self, in_ch=1, out_channel=2):
        super(ResNet, self).__init__()

        self.conv2 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(8)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm3d(64)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm3d(64)
        self.relu7 = nn.ReLU(inplace=True)

        self.maxpool7 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm3d(128)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm3d(128)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm3d(128)
        self.relu10 = nn.ReLU(inplace=True)

        self.maxpool10 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm3d(256)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm3d(256)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm3d(256)
        self.relu13 = nn.ReLU(inplace=True)

        self.maxpool13 = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.relubn1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        #stage 1
        out2 = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))

        #stage 2
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out4 = self.relu4(self.bn4(self.conv4(out3)))
        out4 = out3 + out4
        out4 = self.maxpool4(out4)

        #stage 3
        out5 = self.relu5(self.bn5(self.conv5(out4)))
        out6 = self.relu6(self.bn6(self.conv6(out5)))
        out7 = self.relu7(self.bn7(self.conv7(out6)))
        out7 = out5 + out7
        out7 = self.maxpool7(out7)

        #stage 4
        out8 = self.relu8(self.bn8(self.conv8(out7)))
        out9 = self.relu9(self.bn9(self.conv9(out8)))
        out10 = self.relu10(self.bn10(self.conv10(out9)))
        out10 = out8 + out10
        out10 = self.maxpool10(out10)

        #stage 5
        out11 = self.relu11(self.bn11(self.conv11(out10)))
        out12 = self.relu12(self.bn12(self.conv12(out11)))
        out13 = self.relu13(self.bn13(self.conv13(out12)))
        out13 = out11 + out13
        out13 = self.maxpool13(out13)

        out13 = out13.view(out13.size(0), -1)
        out14 = self.relubn1(self.fcbn1(self.fc1(out13)))
        out15 = self.fc2(out14)
        out_label = F.softmax(out15, dim=1)

        return out_label