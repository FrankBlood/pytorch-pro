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

from utils.utils import load_obj_from_pickle, load_obj_from_json, hyperparam_string

def main():
    # Get the config information from config json data
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to json config",
        required=True
    )
    args = parser.parse_args()
    config_file_path = args.config
    config = load_obj_from_json(config_file_path)

    # Set this experiment name from config information
    experiment_name = hyperparam_string(config)

    save_model_dir = config['data']['save_model_dir']
    load_model_dir = config['data']['load_model_dir']

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='logs/%s' % (experiment_name),
        filemode='w'
    )
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # optional, set the logging level
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


    # Reading Precessed data
    train_set, validation_set, test_set, idx_to_word, word_to_idx = load_obj_from_pickle(curdir+config['data']['dataset'])
    # predictions = len(train_set['source'])

    batch_size = config['data']['batch_size']
    max_length = config['data']['max_src_length']
    vocab_size = len(word_to_idx)


    logging.info('Model Parameters : ')
    logging.info('Task : %s ' % (config['data']['task']))
    logging.info('Model : %s ' % (config['model']['seq2seq']))
    logging.info('Source Language : %s ' % (config['model']['src_lang']))
    logging.info('Target Language : %s ' % (config['model']['trg_lang']))
    logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
    logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
    logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
    logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
    logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
    logging.info('Target RNN Depth : %d ' % (1))
    logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
    logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
    logging.info('Optimizer : %s ' % (config['training']['optimizer']))
    logging.info('Learning Rate : %f ' % (config['training']['lrate']))

    logging.info('Found %d words in src ' % (vocab_size))

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