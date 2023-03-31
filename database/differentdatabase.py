#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：database.py
@Author  ：陈政霖
@Date    ：2023-03-31 11:09
"""

import pandas as pd
from database.databasetemplate import DataBaseTemplate
from const.constant import TABLES, ROOT_PATH, DATA_TYPE, FILER_METHOD


class DataBaseOriginal(DataBaseTemplate):
    pass


class SummaryDataBase(DataBaseTemplate):
    db_root = ROOT_PATH["summary_data"]

    @classmethod
    def read_data_selector(cls, data_type: str, filler_method: str):
        table_name = "summary__{}__{}".format(DATA_TYPE[data_type], FILER_METHOD[filler_method])
        df_result = cls._get_data_from_db(db_name=DATA_TYPE[data_type], table_name=table_name)
        return df_result

    @classmethod
    def read_train_zero_data(cls):
        return cls.read_data_selector(data_type="train_data", filler_method="zero")

    @classmethod
    def read_train_median_data(cls):
        return cls.read_data_selector(data_type="train_data", filler_method="median")

    @classmethod
    def read_train_cluster_median_data(cls):
        return cls.read_data_selector(data_type="train_data", filler_method="cluster_median")

    @classmethod
    def read_val_zero_data(cls):
        return cls.read_data_selector(data_type="val_data", filler_method="zero")

    @classmethod
    def read_val_median_data(cls):
        return cls.read_data_selector(data_type="val_data", filler_method="median")

    @classmethod
    def read_val_cluster_median_data(cls):
        return cls.read_data_selector(data_type="val_data", filler_method="cluster_median")

    @classmethod
    def read_test_zero_data(cls):
        return cls.read_data_selector(data_type="test_data", filler_method="zero")

    @classmethod
    def read_test_median_data(cls):
        return cls.read_data_selector(data_type="test_data", filler_method="median")

    @classmethod
    def read_test_cluster_median_data(cls):
        return cls.read_data_selector(data_type="test_data", filler_method="cluster_median")

    @classmethod
    def store_data_selector(cls, data_type: str, filler_method: str, data: pd.DataFrame):
        table_name = "summary__{}__{}".format(DATA_TYPE[data_type], FILER_METHOD[filler_method])
        cls._store_data_to_db(db_name=DATA_TYPE[data_type], table_name=table_name, data=data)
        pass

    @classmethod
    def stored_train_zero_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="train_data", filler_method="zero", data=data)

    @classmethod
    def stored_train_median_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="train_data", filler_method="median", data=data)

    @classmethod
    def stored_train_cluster_median_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="train_data", filler_method="cluster_median", data=data)

    @classmethod
    def stored_val_zero_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="val_data", filler_method="zero", data=data)

    @classmethod
    def stored_val_median_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="val_data", filler_method="median", data=data)

    @classmethod
    def stored_val_cluster_median_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="val_data", filler_method="cluster_median", data=data)

    @classmethod
    def stored_test_zero_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="test_data", filler_method="zero", data=data)

    @classmethod
    def stored_test_median_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="test_data", filler_method="median", data=data)

    @classmethod
    def stored_test_cluster_median_data(cls, data: pd.DataFrame):
        return cls.store_data_selector(data_type="test_data", filler_method="cluster_median", data=data)


# factors and models:在上述基础上修改数据库名称并加入基金数量即可
class FeaturesDataBase(DataBaseTemplate):
    db_root = ROOT_PATH["features_data"]

    @classmethod
    def read_data_selector(cls, data_type: str, filler_method: str, factor_name: str):
        db_name = "{}__{}".format(DATA_TYPE[data_type], FILER_METHOD[filler_method])
        table_name = TABLES["factor_table"].format(DATA_TYPE[data_type], FILER_METHOD[filler_method], factor_name)
        df_result = cls._get_data_from_db(db_name=db_name, table_name=table_name)
        return df_result

    @classmethod
    def read_train_zero_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="train_data", filler_method="zero", factor_name=factor_name)

    @classmethod
    def read_train_median_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="train_data", filler_method="median", factor_name=factor_name)

    @classmethod
    def read_train_cluster_median_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="train_data", filler_method="cluster_median", factor_name=factor_name)

    @classmethod
    def read_val_zero_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="val_data", filler_method="zero", factor_name=factor_name)

    @classmethod
    def read_val_median_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="val_data", filler_method="median", factor_name=factor_name)

    @classmethod
    def read_val_cluster_median_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="val_data", filler_method="cluster_median", factor_name=factor_name)

    @classmethod
    def read_test_zero_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="test_data", filler_method="zero", factor_name=factor_name)

    @classmethod
    def read_test_median_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="test_data", filler_method="median", factor_name=factor_name)

    @classmethod
    def read_test_cluster_median_data(cls, factor_name: str):
        return cls.read_data_selector(data_type="test_data", filler_method="cluster_median", factor_name=factor_name)

    @classmethod
    def store_data_selector(cls, data_type: str, filler_method: str, factor_name: str, data: pd.DataFrame):
        db_name = "{}__{}".format(DATA_TYPE[data_type], FILER_METHOD[filler_method])
        table_name = TABLES["factor_table"].format(DATA_TYPE[data_type], FILER_METHOD[filler_method], factor_name)
        cls._store_data_to_db(db_name=db_name, table_name=table_name, data=data)
        pass

    @classmethod
    def stored_train_zero_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="train_data", filler_method="zero", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_train_median_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="train_data", filler_method="median", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_train_cluster_median_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="train_data", filler_method="cluster_median", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_val_zero_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="val_data", filler_method="zero", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_val_median_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="val_data", filler_method="median", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_val_cluster_median_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="val_data", filler_method="cluster_median", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_test_zero_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="test_data", filler_method="zero", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_test_median_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="test_data", filler_method="median", data=data,
                                       factor_name=factor_name)

    @classmethod
    def stored_test_cluster_median_data(cls, data: pd.DataFrame, factor_name: str):
        return cls.store_data_selector(data_type="test_data", filler_method="cluster_median", data=data,
                                       factor_name=factor_name)


class ModelsDataBase(DataBaseTemplate):
    db_root = ROOT_PATH["models_data"]

    @classmethod
    def read_data_selector(cls, data_type: str, filler_method: str, year: int, month: int):
        db_name = "{}__{}".format(DATA_TYPE[data_type], FILER_METHOD[filler_method])
        table_name = TABLES["model_table_month"].format(DATA_TYPE[data_type], FILER_METHOD[filler_method], year, month)
        df_result = cls._get_data_from_db(db_name=db_name, table_name=table_name)
        return df_result

    @classmethod
    def read_train_zero_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="train_data", filler_method="zero", year=year, month=month)

    @classmethod
    def read_train_median_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="train_data", filler_method="median", year=year, month=month)

    @classmethod
    def read_train_cluster_median_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="train_data", filler_method="cluster_median", year=year, month=month)

    @classmethod
    def read_val_zero_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="val_data", filler_method="zero", year=year, month=month)

    @classmethod
    def read_val_median_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="val_data", filler_method="median", year=year, month=month)

    @classmethod
    def read_val_cluster_median_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="val_data", filler_method="cluster_median", year=year, month=month)

    @classmethod
    def read_test_zero_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="test_data", filler_method="zero", year=year, month=month)

    @classmethod
    def read_test_median_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="test_data", filler_method="median", year=year, month=month)

    @classmethod
    def read_test_cluster_median_data(cls, year: int, month: int):
        return cls.read_data_selector(data_type="test_data", filler_method="cluster_median", year=year, month=month)

    @classmethod
    def store_data_selector(cls, data_type: str, filler_method: str, year: int, month: int, data: pd.DataFrame):
        db_name = "{}__{}".format(DATA_TYPE[data_type], FILER_METHOD[filler_method])
        table_name = TABLES["model_table_month"].format(DATA_TYPE[data_type], FILER_METHOD[filler_method], year, month)
        cls._store_data_to_db(db_name=db_name, table_name=table_name, data=data)
        pass

    @classmethod
    def stored_train_zero_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="train_data", filler_method="zero", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_train_median_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="train_data", filler_method="median", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_train_cluster_median_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="train_data", filler_method="cluster_median", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_val_zero_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="val_data", filler_method="zero", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_val_median_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="val_data", filler_method="median", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_val_cluster_median_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="val_data", filler_method="cluster_median", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_test_zero_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="test_data", filler_method="zero", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_test_median_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="test_data", filler_method="median", data=data,
                                       year=year, month=month)

    @classmethod
    def stored_test_cluster_median_data(cls, data: pd.DataFrame, year: int, month: int):
        return cls.store_data_selector(data_type="test_data", filler_method="cluster_median", data=data,
                                       year=year, month=month)
