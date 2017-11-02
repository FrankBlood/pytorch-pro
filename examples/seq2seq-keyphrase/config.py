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
print(rootdir+PRO_NAME)
sys.path.insert(0, rootdir + '/Research/' + PRO_NAME)


if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")


DATA_PATH = './data/'