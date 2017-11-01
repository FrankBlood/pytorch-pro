#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DecoderRNN
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-1上午2:07
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

class DecoderRNN(object):
    def __init__(self, output_size, embedding_size, hidden_size, batch_size,
                 max_length, n_layers=1, dropout_p=0.1, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        self.use_cuda = torch.cuda.is_available()

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = F.log_softmax()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, self.batch_size, -1)

        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
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