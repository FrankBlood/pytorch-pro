#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
config
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-1下午7:50
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

from utils import *

PROCESSED_DATA_PATH = './data/processed_data.pkl'

# the max length for this project
MAX_LENGTH = 10

# the eng prefixes for this project
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def read_langs(lang1, lang2, reverse=False):
    '''
    To read the data file we will split the file into lines,
    and then split lines into pairs.
    :param lang1: input sequence
    :param lang2: target sequence
    :param reverse: Boolean for wheather need to be resersed.
    :return: the init Lang obj for both lang and the clean pairs
    '''
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2),
                 encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(p):
    '''
    filte the pair
    :param p: the pair
    :return: Boolean
    '''
    return(len(p[0].split(' ')) < MAX_LENGTH and
           len(p[1].split(' ')) < MAX_LENGTH and
           p[1].startswith(eng_prefixes))


def filter_pairs(pairs):
    '''
    filter all pairs(this is not must)
    :param pairs: all pairs
    :return: filtered pairs
    '''
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    '''
    get the final pairs and the two stat class
    :param lang1: this is the stat class for input data
    :param lang2: this is the stat class for target data
    :param reverse: Boolean
    :return: the two stated Lang class and the filtered pairs
    '''
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def save_processed_data(save_path):
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    save_obj_to_pickle((input_lang, output_lang, pairs), save_path)

if __name__ == '__main__':
    save_processed_data(PROCESSED_DATA_PATH)
