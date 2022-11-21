import torch
import torch.nn as nn
from torch.autograd import Variable

class Focalloss(nn.Module):
    def __init__(self,
                 class_num,
                 alpha=None,
                 gamma=2,
                 size_average=True):
        super(Focalloss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) / class_num
        else:
            assert isinstance(alpha, list) and len(alpha) == class_num
            self.alpha = torch.FloatTensor(alpha)
            self.alpha = self.alpha.view(-1, self.alpha.shape[-1])
            self.alpha = self.alpha.squeeze(1)
            self.alpha = self.alpha / self.alpha.sum()
        # self.alpha = self.alpha.cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        # self.one_hot_codes = torch.eye(self.class_num).cuda()
        self.one_hot_codes = torch.eye(self.class_num)

    def forward(self, input, target):
        assert input.dim() == 2 or input.dim() == 5
        if input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input = input.view(input.numel() // self.class_num, self.class_num)
        target = target.long().view(-1)
        mask = self.one_hot_codes[target.data].cuda()
        mask = Variable(mask, requires_grad=False)
        alpha = self.alpha[target.data].unsqueeze(dim=1).cuda()
        alpha = Variable(alpha, requires_grad=False)
        probs = (input * mask).sum(1).view(-1, 1) + 1e-10  # in order to avoid inf occurs.
        log_probs = probs.log()
        if self.gamma > 0:
            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_probs
        else:
            batch_loss = -alpha * log_probs
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss