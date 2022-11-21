import torch.nn as nn
import torch.nn.functional as F

class FiveCLF(nn.Module):
    def __init__(self, in_ch=(160*15 + 2), out_channel=5):
        super(FiveCLF, self).__init__()

        self.linear1 = nn.Linear(in_ch, 512)  # 128
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(512, 128)  # 32
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear5 = nn.Linear(32, out_channel)



    def forward(self, x):
        out1 = F.dropout(self.linear1(x), p=0.5)
        out1 = self.relu1(self.bn1(out1))
        out2 = F.dropout(self.linear2(out1), p=0.5)
        out2 = self.relu2(self.bn2(out2))
        out3 = F.dropout(self.linear3(out2), p=0.5)
        out3 = self.relu3(self.bn3(out3))
        out4 = F.dropout(self.linear4(out3), p=0.5)
        out4 = self.relu4(self.bn4(out4))
        out5 = self.linear5(out4)
        out_label = F.softmax(out5, dim=1)

        return out_label

# class FiveCLF(nn.Module):
#     def __init__(self, in_ch=5(160*15 + 2), out_channel=5):
#         super(FiveCLF, self).__init__()
#
#         self.linear1 = nn.Linear(in_ch, 160)  # 128
#         self.bn1 = nn.BatchNorm1d(160)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.linear2 = nn.Linear(160, out_channel)
#
#     def forward(self, x):
#         out1 = F.dropout(self.linear1(x), p=0.5)
#         out1 = self.relu1(self.bn1(out1))
#         out1 = self.linear2(out1)
#         out_label = F.softmax(out1, dim=1)
#
#         return out_label
