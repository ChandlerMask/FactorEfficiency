#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：differentfeatures.py
@Author  ：陈政霖
@Date    ：2023-04-01 0:22
"""

import pandas as pd
from features.featurestemplate import FeatureTemplate


class YieldFeature(FeatureTemplate):
    feature_type = "yield"

    def cal_features(self):
        time_list = self._get_time_list()
        df_month_list = self._pick_data_month()
        list_yield = [series.mean() for series in df_month_list]

        df_result_ = pd.DataFrame(index=time_list, data=list_yield)

        df_lag_list = [df_result_.shift(i) for i in range(self.lag+1)]  # 滞后生成变量
        columns_list = ["yield_lag_{}".format(i) for i in range(self.lag+1)]
        df_result = pd.concat(df_lag_list, axis=1)
        df_result.columns = columns_list
        # df_result["date"] = df_result.index
        return df_result


class StdFeature(FeatureTemplate):
    feature_type = "std"

    def cal_features(self):
        time_list = self._get_time_list()
        df_month_list = self._pick_data_month()
        list_yield = [series.std()/series.shape[0] for series in df_month_list]

        df_result_ = pd.DataFrame(index=time_list, data=list_yield)

        df_lag_list = [df_result_.shift(i) for i in range(self.lag+1)]  # 滞后生成变量
        columns_list = ["std_lag_{}".format(i) for i in range(self.lag+1)]
        df_result = pd.concat(df_lag_list, axis=1)
        df_result.columns = columns_list
        # df_result["date"] = df_result.index
        return df_result
