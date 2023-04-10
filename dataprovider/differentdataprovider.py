#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：dataprovider.py
@Author  ：陈政霖
@Date    ：2023-04-09 0:36
"""
import pandas as pd
import numpy as np
import datetime

from database import DataBaseOriginal, SummaryDataBase, FeaturesDataBase, ModelsDataBase
from dataprovider.dataprovidertemplate import DataProviderTemplate


class OlsDataProvider(DataProviderTemplate):

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


class SharpeDataProvider(DataProviderTemplate):

    def _get_next_month(self, date: datetime.date):
        date_new = date + datetime.timedelta(31)
        return datetime.date(year=date_new.year, month=date_new.month, day=1)

    def _get_time_sreies(self, search_data: pd.DataFrame, factor_name: str, s_date: datetime.date):
        e_date = self._get_next_month(s_date)
        result_ = search_data[np.logical_and(search_data["date"] >= s_date, search_data["date"] < e_date)][factor_name]
        result = result_.reset_index(drop=True)
        result.fillna(0, inplace=True)  # 不同月份交易日数量不同，最后的空值空值为0，不影响sharpe率计算
        return result

    def get_time_series(self):
        if self.data is None:
            self._get_all_data()

        data_time_series = eval("SummaryDataBase.read_{}_{}_data".format(self.data_type, self.fill_method))()
        data_time_series["date"] = pd.to_datetime(data_time_series["date"]).dt.date

        model_data = self.data.copy()
        model_data["date"] = pd.to_datetime(model_data["date"]).dt.date

        result_list = [self._get_time_sreies(search_data=data_time_series,
                                             factor_name=model_data.iloc[i, :]["factor_name"],
                                             s_date=model_data.iloc[i, :]["date"])
                       for i in range(model_data.shape[0])]

        time_series = pd.concat(result_list, axis=1)
        return time_series

    def _get_shuffle_index(self, data: pd.DataFrame):
        index_list = list(range(data.shape[0]))
        np.random.shuffle(index_list)
        return index_list

    def get_batch_data(self, batch_size: int):
        features_data = self.get_features_data()
        time_series_data = self.get_yield_dummy()

        index_list = self._get_shuffle_index(features_data)
        for i in range(batch_size, len(index_list), batch_size):
            choosed_index = index_list[i - batch_size: i]
            yield features_data.iloc[choosed_index, :], time_series_data.iloc[:, choosed_index]


class RatioDataProvider(DataProviderTemplate):

    def get_yield_dummy(self):
        if self.data is None:
            self._get_all_data()

        yield_data = self.data["yield_lag_0"]
        yield_data_dummy = np.where(yield_data > 0, 0, 1)
        return yield_data_dummy

    def _get_shuffle_index(self, data: pd.DataFrame):
        index_list = list(range(data.shape[0]))
        np.random.shuffle(index_list)
        return index_list

    def get_batch_data(self, batch_size: int):
        features_data = self.get_features_data()
        yield_dummy_data = self.get_yield_dummy()

        index_list = self._get_shuffle_index(features_data)
        for i in range(batch_size, len(index_list), batch_size):
            choosed_index = index_list[i - batch_size: i]
            yield features_data.iloc[choosed_index, :], yield_dummy_data[choosed_index]
