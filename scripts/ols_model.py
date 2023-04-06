#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：ols_model.py
@Author  ：陈政霖
@Date    ：2023-04-04 17:32 
'''

import pandas as pd
import numpy as np
import datetime
from ols import BestSubsetOls
from database import DataBaseOriginal, SummaryDataBase, FeaturesDataBase, ModelsDataBase
from features import FillerWithZero, FillerWithMedian, FillerWithClusterMedian, YieldFeature, StdFeature
from const.constant import POINTS_DICT, DATA_TYPES_LIST, TARGET_LIST, DROP_LIST
from ols import Ols


class ModelDataProvider(object):

    def __init__(self, time_dict: dict, data_type: str, fill_method: str):
        self.time_dict = time_dict
        self.data_type = data_type
        self.fill_method = fill_method
        self.data = None
        pass

    def _get_all_data(self):
        data_list = []
        for year in self.time_dict:
            for month in self.time_dict[year]:
                data_list.append(
                    eval("ModelsDataBase.read_{}_{}_data".format(self.data_type, self.fill_method))(year=year,
                                                                                                    month=month))
        self.data = pd.concat(data_list, axis=0)
        pass

    def get_features_data(self):
        if self.data is None:
            self._get_all_data()

        columns_list = [column for column in self.data.columns if
                        (column not in DROP_LIST) and (column not in TARGET_LIST)]
        features_data = self.data[columns_list]
        return features_data

    def get_yield_data(self):
        if self.data is None:
            self._get_all_data()

        yield_data = self.data["yield_lag_0"]
        return yield_data

    def get_std_data(self):
        if self.data is None:
            self._get_all_data()

        std_data = self.data["yield_lag_0"]
        return std_data


if __name__ == "__main__":
    time_dict = {2019: [8, 9]}
    data_type = "train"
    fill_method = "median"

    dataprovider = ModelDataProvider(time_dict=time_dict, data_type=data_type, fill_method=fill_method)
    feature_data = dataprovider.get_features_data()
    yield_data = dataprovider.get_yield_data()

    ols = Ols(x=feature_data, y=yield_data)
    ols.ols()
    ols.cal_R()


    y_predict = ols.y_predict
    y = ols.y

    y = self.y
    y_bar = self.y_bar
    y_predict = self.y_predict

    y_predict_tilde = y_predict - y_bar
    y_tilde = y - y_bar

    sse = y_predict_tilde.dot(y_predict_tilde)
    sst = y_tilde.dot(y_tilde)