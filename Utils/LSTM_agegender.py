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
        # self.add_module('AgeGender_interaction_layer', nn.Linear(2, 4, bias=True))
        # self.add_module('Agegender_bn', nn.BatchNorm1d(4))
        # self.add_module('Agegender_LeakyReLU', nn.LeakyReLU(inplace=True))
        self.add_module('FC_layer', nn.Linear(162, 2, bias=True))

    def forward(self, inputs, agegender):
        out1, _ = self.LSTM_layer(inputs)
        out1 = out1.contiguous().view(out1.size(0), -1)
        # print(out.shape)
        # out = out[:, -1, :]
        out1 = self.LeakReLU_layer(self.BN_layer(out1))
        out = torch.cat((out1, agegender), dim=1)
        out = self.FC_layer(out)

        out = F.softmax(out, dim=1)

        return out