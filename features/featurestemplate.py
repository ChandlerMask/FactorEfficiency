#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：featurestemplate.py
@Author  ：陈政霖
@Date    ：2023-03-31 22:46 
'''
import pandas as pd
import datetime
from abc import abstractmethod


class FeatureTemplate(object):
    feature_type = None

    def __init__(self, series: pd.Series, lag: int):
        self.data = series
        self.lag = lag

    def print_feature_type(self):
        print(self.feature_type)

    def _get_time_list(self):
        index_list = list(pd.to_datetime(self.data.index))
        time_index = [datetime.date(year=index.year, month=index.month, day=1) for index in index_list]
        time_list = list(set(time_index))
        time_list.sort()
        return time_list

    def _get_next_month(self, date: datetime.date):
        date_new = date + datetime.timedelta(31)
        return datetime.date(year=date_new.year, month=date_new.month, day=1)

    def _pick_data_month(self):
        data = self.data
        time_list = self._get_time_list()
        month_number = len(time_list)
        df_month_list = [data[time_list[i]: self._get_next_month(time_list[i])] for i in range(month_number)]
        return df_month_list

    @abstractmethod
    def cal_features(self):
        pass
