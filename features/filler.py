#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：filler.py
@Author  ：陈政霖
@Date    ：2023-03-30 16:33 
'''
from abc import ABC

import pandas as pd
import numpy as np
from features.filllertemplate import FillerTemplate
from database import DataBase


class FillerWithZero(FillerTemplate):
    method = "fill with zero"

    def _filler(self):
        data = self.data
        data = data.fillna(0)
        return data


class FillerWithALLMedian(FillerTemplate):
    method = "fill with the median of all factors"

    def _fill_row(self, row: pd.Series) -> pd.Series:
        median = row.median()
        row.fillna(median, inplace=True)
        return row

    def _filler(self):
        data = self.data
        data.apply(lambda x: self._fill_row(x), axis=1)
        return data


cluster_dict = DataBase.get_json()


class FillerWithClusterMedian(FillerTemplate):
    method = "fill with the median in the cluster"
    cluster_dict = cluster_dict

    def _pick_cluster(self):
        for cluster in self.cluster_dict:
            yield self.data[cluster_dict[cluster]].copy()

    def _fill_row(self, row: pd.Series) -> pd.Series:
        median = row.median()
        row.fillna(median, inplace=True)
        return row

    def _filler(self):
        columns = self.data.columns
        result_list = [self._fill_row(df) for df in self._pick_cluster()]
        df_result = pd.concat(result_list, axis=1)
        df_result = df_result[columns]  # 保证因子的顺序与输入时一致
        return df_result


