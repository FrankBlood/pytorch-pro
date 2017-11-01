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