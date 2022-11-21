import torch.nn as nn

class REBNCONV(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv3d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm3d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        xout = self.relu_s1(self.bn_s1(self.conv_s1(x)))

        return xout