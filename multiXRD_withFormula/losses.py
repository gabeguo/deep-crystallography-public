import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

MEAN = 'mean'
SUM = 'sum'

"""
Returns negative log-probability that sample comes from 
multivariate Gaussian distribution with mean zero and user-chosen sigma.
Lower is better (minimize negative probability)
"""
class GaussianDistributionLoss(nn.Module):
    def __init__(self, sigma=0.01, reduction=MEAN):
        super(GaussianDistributionLoss, self).__init__()
        self.sigma = sigma
        self.reduction = reduction
        return
    def forward(self, x):
        batch_size = x.shape[0]
        assert len(x.shape) == 2
        # assume x.shape[0] is batch, x.shape[1] is dimension
        result = 1 / (self.sigma ** 2) * torch.sum(torch.pow(x, 2), dim=1)
        assert result.shape[0] == batch_size

        if self.reduction == MEAN:
            result = torch.mean(result)
        elif self.reduction == SUM:
            result = torch.sum(result)

        return result