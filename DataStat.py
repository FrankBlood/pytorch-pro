#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataStat
======

This is a class

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-10-31下午8:25
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


class DataStat(object):
    '''
    This class will process the clean data set
    which has removed the stop words and useless punctuations.
    '''
    def __init__(self, name):
        '''

        :param name: The name of the data set
        '''
        print("Data stat...")
        self.name = name
        self.word_to_idx = {}
        self.idx_to_word = {0: "SOS", 1: "EOS"}
        self.word_to_count = {}
        self.n_words = 2

    def add_sentence(self, sentence):
        '''

        :param sentence: The cleaned the sentence
        :return: update the param of the class of the entire sentence
        '''
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word):
        '''

        :param word: the word which want to be updated
        :return: the updated param
        '''
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.n_words
            self.idx_to_word[self.n_words] = word
            self.word_to_count[word] = 1
            self.n_words += 1
        else:
            self.word_to_count[word] += 1

    def save_all(self, save_path):
        save_obj_to_pickle((self.word_to_idx, self.idx_to_word, self.word_to_count, self.n_words),
                            save_path)


    def load_all(self, save_path):
        self.word_to_idx, self.idx_to_word, self.word_to_count, self.n_words = load_obj_from_pickle(save_path)


def func():
    pass


if __name__ == "__main__":
    pass