#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
AttnDecoderRNN
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-1上午1:42
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


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, max_length,
                 batch_size=None, n_layers=1, dropout_p=0.1, bidirectional=False):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        self.use_cuda = torch.cuda.is_available()

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attn = nn.Linear(self.hidden_size+self.embedding_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size+self.embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(len(input), 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))

        return output, hidden, attn_weights

    def init_hidden(self):
        # Also, why
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


def func():
    pass


if __name__ == "__main__":
    pass