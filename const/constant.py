#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：constant.py
@Author  ：陈政霖
@Date    ：2023-03-31 10:28 
'''

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
