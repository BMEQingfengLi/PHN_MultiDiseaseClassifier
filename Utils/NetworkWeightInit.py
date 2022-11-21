import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv_s1') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()