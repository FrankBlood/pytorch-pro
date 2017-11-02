#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
run
======

Main script to run the project

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-2上午1:16
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


import logging
import argparse
import numpy as np

from utils import load_obj_from_pickle, load_obj_from_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to json config",
        required=True
    )
    args = parser.parse_args()
    config_file_path = args.config
    config = load_obj_from_json(config_file_path)

    train_set, validation_set, test_set, idx_to_word, word_to_idx = load_obj_from_pickle(curdir+config['data']['dataset'])
    predictions = len(train_set['source'])

    train_data_plain = list(zip(*(train_set['source'], train_set['target'])))
    train_data_source = np.array(train_set['source'])
    train_data_target = np.array(train_set['target'])
    print(len(train_data_plain))
    print(train_data_source.shape)
    print(len(train_data_source[0]))
    print(len(train_data_source[100]))
    print(train_data_target.shape)
    print(len(train_data_target[0]))
    print(len(train_data_target[100]))



if __name__ == '__main__':
    main()