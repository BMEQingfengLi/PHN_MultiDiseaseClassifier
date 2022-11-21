import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMclassifier(nn.Module):
    def __init__(self):
        super(LSTMclassifier, self).__init__()
        self.add_module('LSTM_layer', nn.LSTM(input_size=256,
                                                  hidden_size=16,
                                                  num_layers=2,
                                                  bias=True,
                                                  dropout=0.5,
                                                  batch_first=True,
                                                  bidirectional=False))
        self.add_module('BN_layer', nn.BatchNorm1d(160))
        self.add_module('LeakReLU_layer', nn.LeakyReLU(inplace=True))
        self.add_module('FC_layer', nn.Linear(160, 2, bias=True))

    def forward(self, inputs):
        out, _ = self.LSTM_layer(inputs)
        out = out.contiguous().view(out.size(0), -1)
        # print(out.shape)
        # out = out[:, -1, :]
        out = self.LeakReLU_layer(self.BN_layer(out))
        out = self.FC_layer(out)

        out = F.softmax(out, dim=1)

        return out