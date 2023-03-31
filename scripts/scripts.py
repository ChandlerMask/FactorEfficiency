#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：scripts.py
@Author  ：陈政霖
@Date    ：2023-03-31 15:34 
'''

import numpy as py
import pandas as pd

from ols import BestSubsetOls
from database import DataBaseOriginal, FeaturesDataBase, ModelsDataBase

json = DataBaseOriginal.get_json()
train_data = DataBaseOriginal.get_train_data()

train_data.isnan()