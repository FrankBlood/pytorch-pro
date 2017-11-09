#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
StackedAttentionLSTM
======

Deep Attention LSTM

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-9上午9:09
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
import numpy as np

from LSTMAttentionDot import LSTMAttentionDot

class StackedAttentionLSTM(nn.Module):
    def __init__(self,
            input_size,
            rnn_size,
            num_layers,
            batch_first=True,
            dropout=0.
    ):

        """Initialize params."""
        super(StackedAttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.batch_first = batch_first

        self.layers = []
        for i in range(num_layers):
            layer = LSTMAttentionDot(input_size, rnn_size, batch_first=self.batch_first)
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the layer."""
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            if ctx_mask is not None:
                ctx_mask = torch.ByteTensor(
                    ctx_mask.data.cpu().numpy().astype(np.int32).tolist()
                ).cuda()
            output, (h_1_i, c_1_i) = layer(input, (h_0, c_0), ctx, ctx_mask)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        return input, (h_1, c_1)


def func():
    pass


if __name__ == "__main__":
    pass