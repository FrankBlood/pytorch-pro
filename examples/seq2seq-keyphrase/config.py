#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
config.py
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-2上午1:21
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

import numpy as np

from utils.utils import load_obj_from_pickle, save_obj_to_pickle
from utils.utils import split_into_multiple_and_padding, unk_filter

DATA_PATH = './data/'

def process_data(data, vob_size):
    data_source = np.array(data['source'])
    data_target = np.array(data['target'])

    data_source, data_target = split_into_multiple_and_padding(data_source, data_target)
    data_source, data_target = unk_filter(data_source, vob_size), unk_filter(data_target, vob_size)
    return data_source, data_target


def prepare_train_val_data(metadata_path, vob_size, train_path, val_path, test_path, word_stat_path):

    train_set, validation_set, test_set, idx_to_word, word_to_idx = load_obj_from_pickle(metadata_path)
    print('sample of train_set source', train_set['source'][0])
    print('sample of train_set target', train_set['target'][0])
    print('sample of val_set source', validation_set['source'][0])
    print('sample of val_set target', validation_set['target'][0])
    print('sample of test_set', type(test_set))
    for i in test_set['kdd']:
        print(i)
    print('sample of word to idx', word_to_idx['model'])
    print('sample of idx to word', idx_to_word[2])
    print('sample of idx to word', idx_to_word[1])
    print('sample of idx to word', idx_to_word[0])
    print('word_to_idx', len(word_to_idx))
    print('idx_to_word', len(idx_to_word))
    new_train_set, new_validation_set = {}, {}
    new_train_set['source'], new_train_set['target'] = process_data(train_set, vob_size)
    print("process sucessfully.")
    new_validation_set['source'], new_validation_set['target'] = process_data(validation_set, vob_size)
    print("process successfully.")
    save_obj_to_pickle(test_set, test_path)
    print("save test data successfully.")
    save_obj_to_pickle((idx_to_word, word_to_idx), word_stat_path)
    print("save word stat successfully.")
    save_obj_to_pickle(new_validation_set, val_path)
    print("save val data successfully.")
    save_obj_to_pickle(new_train_set, train_path)
    print("save train data successfully.")


def run_prepare_train_val_data():
    # do nothing is very OK.
    metadata_path = './data/all_600k_dataset.pkl'
    vob_size = 50000
    train_path = './data/train_dataset_processed.pkl'
    val_path = './data/val_dataset_processed.pkl'
    test_path = './data/test_dataset_no_processed.pkl'
    word_stat_path = './data/word_stat.pkl'
    prepare_train_val_data(metadata_path, vob_size, train_path, val_path, test_path, word_stat_path)


if __name__ == '__main__':
    pass
    # run_prepare_train_val_data()

