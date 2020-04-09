# -*-coding: utf-8 -*-
"""
    @Project: torch-Face-Recognize-Pipeline
    @File   : PReLu.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-12-25 18:21:20
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


from torch.nn import PReLU
class PReLU(nn.Module):
    '''
    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input):
        return F.prelu(input, self.weight)

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)
    '''

    def __init__(self, num_parameters=1, init=0.25, inplace=True):
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        self.inplace = inplace

    def forward(self, input):
        res = F.relu(input)
        min_res = F.relu(-input)
        weight_broadcast = self.weight.reshape(1, self.weight.shape[0], 1, 1)
        # print(res.size())
        # print(weight_broadcast.size())
        # print(min_res.size())
        return res - weight_broadcast * min_res

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)
