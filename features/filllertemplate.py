#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：filllertemplate.py
@Author  ：陈政霖
@Date    ：2023-03-30 16:12 
'''

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class FillerTemplate(object):
    method = None

    def __init__(self, data: pd.DataFrame):
        """
        :param data:传入待填充数据
        """
        self.data = data
        pass

    def get_fill_method(self):
        print(self.method)

    def get_time_label(self) -> pd.DataFrame:
        """
        用于确定第一个非空值位置：第一个非空值以前为0，以后为1
        """
        data = self.data
        time_df_ = data.applymap(lambda x: 0 if x is np.nan else 1)
        time_df = time_df_.cumsum(axis=0)  # 通过累加判断当日以前有多少个非空值，用于判断是否有非空值
        result_df = time_df.applymap(lambda x: np.nan if x == 0 else 1)
        return result_df

    @abstractmethod
    def _filler(self):
        pass

    def filler(self):
        time_df = self.get_time_label()
        filler_df = self._filler()
        result_df = time_df * filler_df
        return result_df
