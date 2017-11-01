#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataLoader
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-10-31下午11:24
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

from DataStat import DataStat
from utils import *

class DataLoader(object):
    def __init__(self):
        self.data_stat = DataStat('acdemic')

    def prepare_data(self, input_path, output_path):
        pass


def func():
    pass


if __name__ == "__main__":
    pass