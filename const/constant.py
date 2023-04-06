#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：constant.py
@Author  ：陈政霖
@Date    ：2023-03-31 10:28 
'''

import pandas as pd
import datetime
# 用于检索数据表的名称
TABLES = {
    "factor_table": "features__{}__{}__{}",  # 分别填入数据类型、填充方式和factor
    "model_table_month": "models__{}__{}__{}__{}"  # 分别填入数据类型、填充方式、年、月
}

ROOT_PATH = {"summary_data": "data/dealed_data_summary",
             "features_data": "data/dealed_data_features",
             "models_data": "data/dealed_data_models"}

DATA_TYPE = {"train_data": "train_data",
             "val_data": "val_data",
             "test_data": "test_data"}

FILER_METHOD = {"zero": "zero",
                "median": "median",
                "cluster_median": "cluster_median"}

POINTS_DICT = {"train_data": pd.to_datetime(datetime.date(year=2015, month=1, day=1)),
               "val_data": pd.to_datetime(datetime.date(year=2021, month=1, day=1)),
               "test_data": pd.to_datetime(datetime.date(year=2022, month=8, day=1))}

DATA_TYPES_LIST = ["train", "val", "test"]

TARGET_LIST = ["yield_lag_0", "std_lag_0"]

DROP_LIST = ["date", "factor_name"]