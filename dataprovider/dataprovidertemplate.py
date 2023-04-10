#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：dataprovidertemplate.py
@Author  ：陈政霖
@Date    ：2023-04-09 0:17
"""

import torch
import pandas as pd
import numpy as np
from abc import abstractmethod

from database import DataBaseOriginal, SummaryDataBase, FeaturesDataBase, ModelsDataBase
from const.constant import POINTS_DICT, DATA_TYPES_LIST, TARGET_LIST, DROP_LIST


class DataProviderTemplate(object):

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
        data_result = pd.concat(data_list, axis=0)
        data_result.reset_index(inplace=True)
        del data_result["index"]

        self.data = data_result
        pass

    def get_features_data(self):
        if self.data is None:
            self._get_all_data()

        columns_list = [column for column in self.data.columns if
                        (column not in DROP_LIST) and (column not in TARGET_LIST)]
        features_data = self.data[columns_list]
        return features_data

