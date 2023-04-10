#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：dataprovider_test.py
@Author  ：陈政霖
@Date    ：2023-04-09 11:23
"""
import numpy as np
from dataprovider import OlsDataProvider, SharpeDataProvider, RatioDataProvider

# setting
time_dict = {2020: [11, 12]}
data_type = "train"
fill_method = "median"
batch_size = 30

ols_provider = OlsDataProvider(time_dict=time_dict, data_type=data_type, fill_method=fill_method)
ols_targets_data = ols_provider.get_features_data()
ols_labels_data = ols_provider.get_yield_data()
ols_labels_data_std = ols_provider.get_std_data()

ratio_provider = RatioDataProvider(time_dict=time_dict, data_type=data_type, fill_method=fill_method)
ratio_targets_data = ratio_provider.get_features_data()
ratio_labels_data = ratio_provider.get_yield_dummy()

ratio_loader = ratio_provider.get_batch_data(batch_size=batch_size)
for x, y in ratio_loader:
    print(x)
    print(y)
    break

np.where(ols_labels_data[list(x.index)] > 0, 0, 1) == y


sharpe_provider = SharpeDataProvider(time_dict=time_dict, data_type=data_type, fill_method=fill_method)
sharpe_features_data = sharpe_provider.get_features_data()
sharpe_time_series = sharpe_provider.get_time_series()

eval("SummaryDataBase.read_{}_{}_data".format(data_type, fill_method))()[["date", "f_007980"]]
