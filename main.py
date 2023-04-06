#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：main.py
@Author  ：陈政霖
@Date    ：2023-03-29 23:21 
'''
import numpy as np
import numpy as py
import pandas as pd
import datetime
from tqdm import tqdm
from ols import BestSubsetOls
from database import DataBaseOriginal, SummaryDataBase, FeaturesDataBase, ModelsDataBase
from features import FillerWithZero, FillerWithMedian, FillerWithClusterMedian, YieldFeature, StdFeature
from const.constant import POINTS_DICT, DATA_TYPES_LIST


def cal_nan(data: pd.DataFrame):
    return data.isnull().sum().sum()


def find_cluster(cluster_dict: dict, factor: str):
    for key, value in cluster_dict.items():
        if factor in value:
            return key


# data processing

# fill data
data_types_list = ["train", "val", "test"]
fill_methods_list = ["zero", "median", "cluster_median"]
class_name_dict = {"zero": "Zero", "median": "Median", "cluster_median": "ClusterMedian"}

for data_type in data_types_list:
    for fill_method in fill_methods_list:
        data = eval("DataBaseOriginal.get_{}_data".format(data_type))()
        fill_tool = eval("FillerWith{}".format(class_name_dict[fill_method]))(data)
        result_df = fill_tool.filler()
        eval("SummaryDataBase.stored_{}_{}_data".format(data_type, fill_method))(result_df)

# for data_type in data_types_list:
#     fill_method = "cluster_median"
#     data = eval("DataBaseOriginal.get_{}_data".format(data_type))()
#     fill_tool = eval("FillerWith{}".format(class_name_dict[fill_method]))(data)
#     result_df = fill_tool.filler()
#     eval("SummaryDataBase.stored_{}_{}_data".format(data_type, fill_method))(result_df)

# calculate features
class DataProcessorGetFeatures(object):
    point_dict = POINTS_DICT
    cluster_dict = DataBaseOriginal.get_json()
    data_types_list = DATA_TYPES_LIST

    def __init__(self, fill_method: str, lag: int):
        self.fill_method = fill_method
        self.data = None
        self.lag = lag
        pass

    def _get_data_all_time(self) -> pd.DataFrame:
        """
        用于将train、val、test数据合并，因为val和test在计算feature的时候会用到train的数据
        :return:
        """
        fill_method = self.fill_method
        data_list = [eval("SummaryDataBase.read_{}_{}_data".format(data_type, fill_method))() for data_type in
                     self.data_types_list]
        data_all = pd.concat(data_list, axis=0)
        return data_all

    def _cal_cluster_features(self) -> dict:
        cluster_dict = self.cluster_dict
        data = self._get_data_all_time()
        data["date"] = pd.to_datetime(data["date"])
        data.index = data["date"]
        del data["date"]

        cluster_feature_dict = {}
        for cluster in cluster_dict:
            df_cluster = data[cluster_dict[cluster]].mean(axis=1)
            feature_cluster_1 = YieldFeature(series=df_cluster, lag=self.lag).cal_features()
            feature_cluster_2 = StdFeature(series=df_cluster, lag=self.lag).cal_features()

            del feature_cluster_1["yield_lag_0"]
            del feature_cluster_2["std_lag_0"]# 删除未来信息
            feature_cluster_1.columns = ["cluster_" + column for column in feature_cluster_1.columns]
            feature_cluster_2.columns = ["cluster_" + column for column in feature_cluster_2.columns]
            cluster_feature_dict[cluster] = pd.concat([feature_cluster_1, feature_cluster_2], axis=1)

        return data, cluster_feature_dict

    def _store_data_to_different_data_type(self, data: pd.DataFrame, factor_name: str):
        point_dict = self.point_dict

        data_train = data[data["date"] < point_dict["val_data"]]
        eval("FeaturesDataBase.stored_train_{}_data".format(self.fill_method))(data=data_train, factor_name=factor_name)

        data_val = data[np.logical_and(point_dict["val_data"] <= data["date"], data["date"] < point_dict["test_data"])]
        eval("FeaturesDataBase.stored_val_{}_data".format(self.fill_method))(data=data_val, factor_name=factor_name)

        data_test = data[point_dict["test_data"] <= data["date"]]
        eval("FeaturesDataBase.stored_test_{}_data".format(self.fill_method))(data=data_test, factor_name=factor_name)

    def _find_cluster(self, cluster_dict: dict, factor: str):
        for key, value in cluster_dict.items():
            if factor in value:
                return key

    def cal_features(self):
        data, cluster_feature_dict = self._cal_cluster_features()

        for factor in data.columns:
            series_factor = data[factor]
            factor_result_1 = YieldFeature(series=series_factor, lag=self.lag).cal_features()
            factor_result_2 = StdFeature(series=series_factor, lag=self.lag).cal_features()
            cluster = self._find_cluster(cluster_dict=self.cluster_dict, factor=factor)
            result = pd.concat([factor_result_1, factor_result_2, cluster_feature_dict[cluster]], axis=1)
            result["date"] = result.index

            self._store_data_to_different_data_type(data=result, factor_name=factor)
        pass


for fill_method in tqdm(fill_methods_list):
    DataProcessorGetFeatures(fill_method, lag=12).cal_features()


# processor = DataProcessorGetFeatures(fill_method="zero", lag=12)
# processor.cal_features()


class DataProcessorGetModelsData(object):
    def __init__(self, data_type: str, fill_method: str):
        self.data_type = data_type
        self.fill_method = fill_method
        pass

    def print_processing_data(self):
        print("the data being processed are {}_{} ".format(self.data_type, self.fill_method))
        pass

    def _get_time_list_from_summary(self, data: pd.DataFrame):
        index_list = list(pd.to_datetime(data["date"]))
        time_index = [datetime.date(year=index.year, month=index.month, day=1) for index in index_list]
        time_list = list(set(time_index))
        time_list.sort()
        return time_list

    def get_time_factors_from_summary(self):
        """
        从summary数据库中获取时间和因子列表
        :return:
        """
        data = eval("SummaryDataBase.read_{}_{}_data".format(self.data_type, self.fill_method))()
        time_list = self._get_time_list_from_summary(data)

        factor_list = list(data.columns)
        factor_list.remove("date")
        return time_list, factor_list

    def get_models_data_from_features(self):
        time_list, factor_list = self.get_time_factors_from_summary()
        data_dict = {}
        for factor in factor_list:
            data_factor = eval("FeaturesDataBase.read_{}_{}_data".format(self.data_type, self.fill_method))(
                factor_name=factor)
            data_factor["date"] = pd.to_datetime(data_factor["date"]).dt.date
            data_dict[factor] = [data_factor[data_factor["date"] == time] for time in time_list]

        for i in range(len(time_list)):
            data_list = [data_dict[factor][i] for factor in factor_list]
            df_result = pd.concat(data_list, axis=0)
            df_result["factor_name"] = factor_list
            year = time_list[i].year
            month = time_list[i].month
            eval("ModelsDataBase.stored_{}_{}_data".format(self.data_type, self.fill_method))(data=df_result, year=year,
                                                                                              month=month)


for data_type in data_types_list:
    for fill_method in fill_methods_list:
        process = DataProcessorGetModelsData(data_type, fill_method)
        process.get_models_data_from_features()

