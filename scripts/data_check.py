#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：data_check.py
@Author  ：陈政霖
@Date    ：2023-04-03 14:09 
'''

import pandas as pd
import numpy as np
import datetime
from ols import BestSubsetOls
from database import DataBaseOriginal, SummaryDataBase, FeaturesDataBase, ModelsDataBase
from features import FillerWithZero, FillerWithMedian, FillerWithClusterMedian, YieldFeature, StdFeature
from const.constant import POINTS_DICT, DATA_TYPES_LIST

cluster_dict = DataBaseOriginal.get_json()

# filled with median
df_filled = SummaryDataBase.read_train_median_data()

df_original = DataBaseOriginal.get_train_data()
df_empty = df_original[df_original.isnull().any(axis=1)]
for i in range(-10, -1):
    df_filled_new = df_empty.iloc[i, :]
    df_filled_new = df_filled_new.fillna(df_filled_new[1:].median())
    date = df_filled_new["date"]
    df_filled_database = df_filled[df_filled["date"] == date]
    print((df_filled_database == df_filled_new).all().all())

# filled with cluster_median
cluster_name = "cluster_00038"
cluster = cluster_dict[cluster_name]

df_filled = SummaryDataBase.read_train_cluster_median_data()
df_cluster_filled = df_filled[cluster]

df_original = DataBaseOriginal.get_train_data()
df_cluster_original = df_original[cluster]
df_empty = df_cluster_original[df_cluster_original.isnull().any(axis=1)]

for i in range(-10, -1):
    df_filled_new = df_empty.iloc[i, :]
    df_filled_new = df_filled_new.fillna(df_filled_new.median())
    index = df_filled_new.name
    df_filled_database = df_cluster_filled.iloc[index]
    print((df_filled_database == df_filled_new).all().all())

# zero
cluster_name = "cluster_00038"
cluster = cluster_dict[cluster_name]

df_filled = SummaryDataBase.read_train_zero_data()
df_cluster_filled = df_filled[cluster]
df_cluster_filled.iloc[1110, :]

# factor


data_original_ = SummaryDataBase.read_train_cluster_median_data()
data_original_["date"] = pd.to_datetime(data_original_["date"])
start_date = pd.to_datetime(datetime.date(2020, 10, 1))
end_date = pd.to_datetime(datetime.date(2020, 11, 1))
data_original = data_original_[np.logical_and(start_date <= data_original_["date"], data_original_["date"] < end_date)]
data_calculate = data_original['f_007723']
mean_yield = data_calculate.mean()
new_std = data_calculate.std() / data_calculate.shape[0]

factor_db = FeaturesDataBase.read_train_cluster_median_data(factor_name='f_007723')
factor_db["date"] = pd.to_datetime(factor_db["date"])
new_yield = factor_db[factor_db["date"] == start_date]["yield_lag_0"]
new_std = factor_db[factor_db["date"] == start_date]["std_lag_0"]

print(mean_yield == new_yield)
print(new_std == new_std)


# model data
years = [2020, 2019]
months = [5, 8]
factors = ['f_007723', 'f_007961']
for year in years:
    for month in months:
        for factor in factors:
            start_date = pd.to_datetime(datetime.date(year, month, 1))

            df_model = ModelsDataBase.read_train_zero_data(year=year, month=month)
            yield_original = df_model[df_model["factor_name"] == factor]["yield_lag_0"].values
            std_original = df_model[df_model["factor_name"] == factor]["std_lag_0"].values

            factor_db = FeaturesDataBase.read_train_cluster_median_data(factor_name=factor)
            factor_db["date"] = pd.to_datetime(factor_db["date"])
            new_yield = factor_db[factor_db["date"] == start_date]["yield_lag_0"]
            new_std = factor_db[factor_db["date"] == start_date]["std_lag_0"]

            print(yield_original == new_yield)
            print(std_original == new_std)