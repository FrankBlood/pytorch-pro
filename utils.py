#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
utils
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-10-31下午9:00
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

import json
import cPickle as pickle
import hickle
from io import open
import unicodedata
import string
import re
import random
import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

use_cuda = torch.cuda.is_available()

# This is the start token and end token's index
SOS_token = 0
EOS_token = 1


class Lang(object):
    '''
    This is the class for get the stat information of the corpus.
    '''
    def __init__(self, name):
        self.name = name
        self.word_to_idx = {}
        self.word_to_count = {}
        self.idx_to_word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.n_words
            self.word_to_count[word] = 1
            self.idx_to_word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_to_count[word] += 1


def unicode_to_Ascii(s):
    '''
    Turn a Unicode string to plain ASCII, thanks to
    http://stackoverflow.com/a/518232/2809427
    :param s: string
    :return: string
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    '''
    Lowercase, trim, and remove non-letter characters
    :param s: string
    :return: normalize the string
    '''
    s = unicode_to_Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def indexes_from_sentence(lang, sentence):
    '''
    Get indexes of the words in the input sentence
    :param lang: the stated class for corpus
    :param sentence: the sentence which will turn to idx sequence
    :return: the idx sequence
    '''
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    '''
    Get torch Variable from sentence
    :param lang: the stated class for corpus
    :param sentence: the sentence: string
    :return: the variable for idx sequence
    '''
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variables_from_pair(input_lang, output_lang, pair):
    '''
    Get the Variable from pair
    :param input_lang: the input stated class
    :param output_lang: the target stated class
    :param pair: the sentence pair
    :return: the variable for the pair
    '''
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


def train_seq2seq(input_variable, target_variable, encoder, decoder,
                  encoder_optimizer, decoder_optimizer, criterion,
                  max_length, teacher_forcing_ratio = 0.5):
    '''
    Train for one pair
    :param input_variable: Variable of input
    :param target_variable: Variable of target
    :param encoder: the encoder class
    :param decoder: the decoder class
    :param encoder_optimizer: the optimizer of encoder
    :param decoder_optimizer: the optimizer of decoder
    :param criterion: the criterion function
    :param max_length: the max length of the sequence
    :param teacher_forcing_ratio: the teacher forcing ratio
    :return: the avage loss of entire target sequence
    '''

    # init hidden state for encoder
    encoder_hidden = encoder.init_hidden()

    # init optimizer
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # get this pair's sentence length
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # all encoder outputs of the input sequence,
    # just like padding for the sequence
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    # do this process for every embedding in sequence
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    # the first input of the decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # the first decoder hidden is the last encoder hidden
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # What is this...
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            # update the decoder input by last output of the decoder
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def train_iters(encoder, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variables_from_pair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train_seq2seq(input_variable, target_variable, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('./imgs/' + '_'.join(time.asctime(time.localtime(time.time())).split(' ')) + '.jpg',
                dpi=1000)


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    plt.savefig('./imgs/' + '_'.join(time.asctime(time.localtime(time.time())).split(' ')) + '.pdf',
                format='pdf')


def evaluate_and_show_attention(encoder, decoder, input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def as_minutes(s):
    '''
    as minutes
    :param s:
    :return: string
    '''
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    '''
    time since
    :param since:
    :param percent:
    :return: string
    '''
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def save_obj_to_json(obj, save_path):
    '''

    :param obj: the name of the param
    :param save_path: the save path
    :return:
    '''
    with open(save_path, 'w') as fw:
        fw.write(json.dumps(obj))


def save_obj_to_pickle(obj, save_path):
    with open(save_path, 'wb') as fw:
        pickle.dump(obj, fw, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj_from_json(save_path):
    '''

    :param save_path: the save path
    :return:
    '''
    with open(save_path, 'r') as fp:
        return json.load(fp)


def load_obj_from_pickle(save_path):
    with open(save_path, 'rb') as fp:
        return pickle.load(fp)