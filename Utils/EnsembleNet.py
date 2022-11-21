import torch.nn as nn
import torch.nn.functional as F

class EnsembleNet(nn.Module):
    def __init__(self, in_ch=256*10, out_channel=2):
        super(EnsembleNet, self).__init__()

        self.linear1 = nn.Linear(in_ch, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(64, out_channel)


    def forward(self, x):
        out1 = F.dropout(self.linear1(x), p=0.5)
        out1 = self.relu1(self.bn1(out1))
        out2 = F.dropout(self.linear2(out1), p=0.5)
        out2 = self.relu2(self.bn2(out2))
        out3 = self.linear3(out2)
        out_label = F.softmax(out3, dim=1)

        return out_label