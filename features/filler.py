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
from database import DataBaseOriginal


class FillerWithZero(FillerTemplate):
    method = "fill with zero"

    def _filler(self):
        data = self.data
        data = data.fillna(0)
        return data


class FillerWithMedian(FillerTemplate):
    method = "fill with the median of all factors"

    def _fill_row(self, row: pd.Series) -> pd.Series:
        if row.index[0] == "date":
            median = row[1:].median()
        else:
            raise KeyError("the first column is not data, please have a check")

        row.fillna(median, inplace=True)
        return row

    def _filler(self):
        data = self.data
        data = data.apply(lambda x: self._fill_row(x), axis=1)
        return data


cluster_dict = DataBaseOriginal.get_json()


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
        date_column = self.data["date"].to_frame("date")
        result_list = [df.apply(lambda x: self._fill_row(x), axis=1) for df in self._pick_cluster()]
        df_result = pd.concat(result_list, axis=1)
        df_result = pd.concat([date_column, df_result], axis=1)
        df_result = df_result[columns]  # 保证因子的顺序与输入时一致
        return df_result


