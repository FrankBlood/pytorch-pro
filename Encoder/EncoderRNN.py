#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
EncoderRNN
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-10-31下午11:35
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size, n_layers=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.use_cuda = torch.cuda.is_available()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(len(input), self.batch_size, -1)
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def init_hidden(self):
        if self.bidirectional:
            result = Variable(torch.zeros(2*self.n_layers, self.batch_size, self.hidden_size))
        else:
            result = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


def func():
    pass


if __name__ == "__main__":
    pass