#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：main.py
@Author  ：陈政霖
@Date    ：2023-03-29 23:21 
'''
import numpy as py
import pandas as pd

from ols import BestSubsetOls
from database import DataBaseOriginal, FeaturesDataBase, ModelsDataBase
from features import FillerWithZero, FillerWithALLMedian, FillerWithClusterMedian

# data processing


