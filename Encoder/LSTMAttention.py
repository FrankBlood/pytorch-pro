#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
LSTMAttention
======

A long short-term memory (LSTM) cell with attention.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-2下午3:33
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
import torch.nn.functional as F
import math


class LSTMAttention(nn.Module):

    def __init__(self, input_size, hidden_size, context_size):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = 1

        self.input_weights_1 = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.hidden_weights_1 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.input_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.input_weights_2 = nn.Parameter(torch.Tensor(4 * hidden_size, context_size))
        self.hidden_weights_2 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.input_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.context2attention = nn.Parameter(torch.Tensor(context_size, context_size))
        self.bias_context2attention = nn.Parameter(torch.Tensor(context_size))

        self.hidden2attention = nn.Parameter(torch.Tensor(context_size, hidden_size))

        self.input2attention = nn.Parameter(torch.Tensor(input_size, context_size))

        self.recurrent2attention = nn.Parameter(torch.Tensor(context_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_ctx = 1.0 / math.sqrt(self.context_size)

        self.input_weights_1.data.uniform_(-stdv, stdv)
        self.hidden_weights_1.data.uniform_(-stdv, stdv)
        self.input_bias_1.data.fill_(0)
        self.hidden_bias_1.data.fill_(0)

        self.input_weights_2.data.uniform_(-stdv_ctx, stdv_ctx)
        self.hidden_weights_2.data.uniform_(-stdv, stdv)
        self.input_bias_2.data.fill_(0)
        self.hidden_bias_2.data.fill_(0)

        self.context2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.bias_context2attention.data.fill_(0)

        self.hidden2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.input2attention.data.uniform_(-stdv_ctx, stdv_ctx)

        self.recurrent2attention.data.uniform_(-stdv_ctx, stdv_ctx)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden, projected_input, projected_ctx):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim

            gates = F.linear(input, self.input_weights_1, self.input_bias_1) + \
                    F.linear(hx, self.hidden_weights_1, self.hidden_bias_1)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # Attention mechanism

            # Project current hidden state to context size
            hidden_ctx = F.linear(hy, self.hidden2attention)

            # Added projected hidden state to each projected context
            hidden_ctx_sum = projected_ctx + hidden_ctx.unsqueeze(0).expand(
                projected_ctx.size()
            )

            # Add this to projected input at this time step
            hidden_ctx_sum = hidden_ctx_sum + \
                projected_input.unsqueeze(0).expand(hidden_ctx_sum.size())

            # Non-linearity
            hidden_ctx_sum = F.tanh(hidden_ctx_sum)

            # Compute alignments
            alpha = torch.bmm(hidden_ctx_sum.transpose(0, 1),
                              self.recurrent2attention.unsqueeze(0).expand(hidden_ctx_sum.size(1),
                                                                           self.recurrent2attention.size(0),
                                                                           self.recurrent2attention.size(1))
                              ).squeeze()
            alpha = F.softmax(alpha)
            weighted_context = torch.mul(ctx,
                                         alpha.t().unsqueeze(2).expand(ctx.size())
                                         ).sum(0).squeeze()

            gates = F.linear(weighted_context, self.input_weights_2, self.input_bias_2) +\
                    F.linear(hy, self.hidden_weights_2, self.hidden_bias_2)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cy) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        input = input.transpose(0, 1)
        projected_ctx = torch.bmm(ctx,
                                  self.context2attention.unsqueeze(0).expand(ctx.size(0),
                                                                             self.context2attention.size(0),
                                                                             self.context2attention.size(1)),
                                  )
        projected_ctx += self.bias_context2attention.unsqueeze(0).unsqueeze(0).expand(projected_ctx.size())

        projected_input = torch.bmm(input,
                                    self.input2attention.unsqueeze(0).expand(input.size(0),
                                                                             self.input2attention.size(0),
                                                                             self.input2attention.size(1)),
                                    )

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, projected_input[i], projected_ctx)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden


def func():
    pass


if __name__ == "__main__":
    pass