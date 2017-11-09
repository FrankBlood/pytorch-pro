#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DeepBidirectionalLSTM
======

A Deep LSTM with the first layer being bidirectional.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-2下午3:00
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:3])
PRO_NAME = 'pytorch-pro'
sys.path.insert(0, rootdir + '/Research/' + PRO_NAME)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

import torch
import torch.nn as nn
from torch.autograd import Variable

class DeepBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 num_layers, dropout, batch_first):

        """Initialize params."""
        super(DeepBidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.use_cuda = torch.cuda.is_available()

        self.bi_encoder = nn.LSTM(
            self.input_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=self.batch_first,
            dropout=self.dropout
        )

        self.encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers - 1,
            bidirectional=False,
            batch_first=self.batch_first,
            dropout=self.dropout
        )

    def forward(self, input):
        """
        Propogate input forward through the network.
        :param input:
        :return:
        """
        hidden_bi, hidden_deep = self.init_hidden(input)
        bilstm_output, (_, _) = self.bi_encoder(input, hidden_bi)
        return self.encoder(bilstm_output, hidden_deep)

    def init_hidden(self, input):
        '''
        Get cell states and hidden states.
        :param input:
        :return:
        '''
        batch_size = input.size(0) if self.encoder.batch_first else input.size(1)
        h0_encoder_bi = Variable(torch.zeros(2, batch_size, self.hidden_size // 2))
        c0_encoder_bi = Variable(torch.zeros(2, batch_size, self.hidden_size // 2))

        h0_encoder = Variable(torch.zeros(self.num_layers - 1, batch_size, self.hidden_size))

        c0_encoder = Variable(torch.zeros(self.num_layers - 1, batch_size, self.hidden_size))

        if self.use_cuda:
            return (h0_encoder_bi.cuda(), c0_encoder_bi.cuda()), (h0_encoder.cuda(), c0_encoder.cuda())
        else:
            return (h0_encoder_bi, c0_encoder_bi), (h0_encoder, c0_encoder)

def func():
    pass


if __name__ == "__main__":
    pass