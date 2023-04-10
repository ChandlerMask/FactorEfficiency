#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：ols_test.py
@Author  ：陈政霖
@Date    ：2023-04-06 23:10
"""

import pandas as pd
import numpy as np
import datetime
from ols import BestSubsetOls
from database import DataBaseOriginal, SummaryDataBase, FeaturesDataBase, ModelsDataBase
from features import FillerWithZero, FillerWithMedian, FillerWithClusterMedian, YieldFeature, StdFeature
from const.constant import POINTS_DICT, DATA_TYPES_LIST, TARGET_LIST, DROP_LIST
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy import linalg

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


class Ols_1(object):
    """
    Ols类用于单次回归并计算R_squared、likelihood、aic、bic指标
    Ols内不进行数据的切割，best_subset和cv通过类外对象完成
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

        self.n = x.shape[0]  # 样本个数
        self.p = x.shape[1] + 1  # 变量个数

        self.x_bar = self.x.mean(axis=0)
        self.y_bar = self.y.mean()

        self.x_center = np.dot(self.x.T, self.x) - self.n * np.outer(self.x_bar, self.x_bar)
        self.xy_center = np.dot(self.x.T, self.y) - self.n * self.y_bar * self.x_bar

        self.beta = None
        self.y_predict = None
        self.log_likelihood = None
        pass

    def ols_1(self):
        print("new")
        b = linalg.solve(self.x_center, self.xy_center, assume_a='pos')
        b0 = self.y_bar - np.inner(self.x_bar, b)
        # L = linalg.cholesky(self.x_center)
        # b, _ = linalg.lapack.dpotrs(L, self.xy_center)
        # b0 = self.y_bar - np.inner(self.x_bar, b)
        beta = np.r_[b0, b]
        return beta

    def cal_predict(self, x: np.ndarray = None):
        if self.beta is None:
            self.beta = self.ols_1()
        beta = self.beta

        if isinstance(x, np.ndarray):
            x_ = x
        else:
            x_ = self.x

        x = sm.add_constant(x_)
        y_predict = x.dot(beta)

        return  beta, x, y_predict

    def cal_R(self):
        if self.y_predict is None:
            self.y_predict = self.cal_predict()

        y = self.y
        y_bar = self.y_bar
        y_predict = self.y_predict

        y_predict_tilde = y_predict - y_bar
        y_tilde = y - y_bar

        sse = y_predict_tilde.dot(y_predict_tilde)
        sst = y_tilde.dot(y_tilde)
        return sse / sst

    def cal_likelihood(self):
        if self.y_predict is None:
            self.y_predict = self.cal_predict()

        y_predict = self.y_predict
        y = self.y
        n = self.n

        # 估算总体方差
        ssr_ = y_predict - y
        ssr = ssr_.dot(ssr_)
        sigma = np.sqrt(ssr / n)
        log_likelihood = np.sum(norm.logpdf(y, loc=y_predict, scale=sigma))
        return log_likelihood

    def cal_aic(self):
        if self.log_likelihood is None:
            self.log_likelihood = self.cal_likelihood()

        log_likelihood = self.log_likelihood
        p = self.p
        aic = -2 * log_likelihood + 2 * p
        return aic

    def cal_bic(self):
        if self.log_likelihood is None:
            self.log_likelihood = self.cal_likelihood()

        log_likelihood = self.log_likelihood
        p = self.p
        n = self.n
        bic = -2 * log_likelihood + np.log(n) * p

        return bic

class Ols_2(object):
    """
    Ols类用于单次回归并计算R_squared、likelihood、aic、bic指标
    Ols内不进行数据的切割，best_subset和cv通过类外对象完成
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

        self.n = x.shape[0]  # 样本个数
        self.p = x.shape[1] + 1  # 变量个数

        self.x_bar = self.x.mean(axis=0)
        self.y_bar = self.y.mean()

        self.x_center = np.dot(self.x.T, self.x) - self.n * np.outer(self.x_bar, self.x_bar)
        self.xy_center = np.dot(self.x.T, self.y) - self.n * self.y_bar * self.x_bar

        self.beta = None
        self.y_predict = None
        self.log_likelihood = None
        pass

    def ols_2(self):
        print("3")
        L = linalg.cholesky(self.x_center)
        b, _ = linalg.lapack.dpotrs(L, self.xy_center)
        b0 = self.y_bar - np.inner(self.x_bar, b)
        beta = np.r_[b0, b]
        return beta

    def cal_predict(self, x: np.ndarray = None):
        if self.beta is None:
            self.beta = self.ols_2()
        beta = self.beta

        if isinstance(x, np.ndarray):
            x_ = x
        else:
            x_ = self.x

        x = sm.add_constant(x_)
        y_predict = x.dot(beta)

        return beta, x, y_predict

    def cal_R(self):
        if self.y_predict is None:
            _, _, self.y_predict = self.cal_predict()

        y = self.y
        y_bar = self.y_bar
        y_predict = self.y_predict

        y_predict_tilde = y_predict - y_bar
        y_tilde = y - y_bar

        sse = y_predict_tilde.dot(y_predict_tilde)
        sst = y_tilde.dot(y_tilde)
        return sse / sst

    def cal_likelihood(self):
        if self.y_predict is None:
            self.y_predict = self.cal_predict()

        y_predict = self.y_predict
        y = self.y
        n = self.n

        # 估算总体方差
        ssr_ = y_predict - y
        ssr = ssr_.dot(ssr_)
        sigma = np.sqrt(ssr / n)
        log_likelihood = np.sum(norm.logpdf(y, loc=y_predict, scale=sigma))
        return log_likelihood

    def cal_aic(self):
        if self.log_likelihood is None:
            self.log_likelihood = self.cal_likelihood()

        log_likelihood = self.log_likelihood
        p = self.p
        aic = -2 * log_likelihood + 2 * p
        return aic

    def cal_bic(self):
        if self.log_likelihood is None:
            self.log_likelihood = self.cal_likelihood()

        log_likelihood = self.log_likelihood
        p = self.p
        n = self.n
        bic = -2 * log_likelihood + np.log(n) * p

        return bic

class Ols(object):
    """
    Ols类用于单次回归并计算R_squared、likelihood、aic、bic指标
    Ols内不进行数据的切割，best_subset和cv通过类外对象完成
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

        self.n = x.shape[0]  # 样本个数
        self.p = x.shape[1] + 1  # 变量个数

        self.x_bar = self.x.mean(axis=0)
        self.y_bar = self.y.mean()

        self.x_center = np.dot(self.x.T, self.x) - self.n * np.outer(self.x_bar, self.x_bar)
        self.xy_center = np.dot(self.x.T, self.y) - self.n * self.y_bar * self.x_bar

        self.beta = None
        self.y_predict = None
        self.log_likelihood = None
        pass

    def ols(self):
        print("new")
        L = linalg.cholesky(self.x_center)
        b, _ = linalg.lapack.dpotrs(L, self.xy_center)
        b0 = self.y_bar - np.inner(self.x_bar, b)
        beta = np.r_[b0, b]
        return beta

    def cal_predict(self, x: np.ndarray = None):
        if self.beta is None:
            self.beta = self.ols()
        beta = self.beta

        if isinstance(x, np.ndarray):
            x_ = x
        else:
            x_ = self.x

        x = sm.add_constant(x_)
        y_predict = x.dot(beta)

        return y_predict

    def cal_R(self):
        if self.y_predict is None:
            self.y_predict = self.cal_predict()

        y = self.y
        y_bar = self.y_bar
        y_predict = self.y_predict

        y_predict_tilde = y_predict - y_bar
        y_tilde = y - y_bar

        sse = y_predict_tilde.dot(y_predict_tilde)
        sst = y_tilde.dot(y_tilde)
        return sse / sst

    def cal_likelihood(self):
        if self.y_predict is None:
            self.y_predict = self.cal_predict()

        y_predict = self.y_predict
        y = self.y
        n = self.n

        # 估算总体方差
        ssr_ = y_predict - y
        ssr = ssr_.dot(ssr_)
        sigma = np.sqrt(ssr / n)
        log_likelihood = np.sum(norm.logpdf(y, loc=y_predict, scale=sigma))
        return log_likelihood

    def cal_aic(self):
        if self.log_likelihood is None:
            self.log_likelihood = self.cal_likelihood()

        log_likelihood = self.log_likelihood
        p = self.p
        aic = -2 * log_likelihood + 2 * p
        return aic

    def cal_bic(self):
        if self.log_likelihood is None:
            self.log_likelihood = self.cal_likelihood()

        log_likelihood = self.log_likelihood
        p = self.p
        n = self.n
        bic = -2 * log_likelihood + np.log(n) * p

        return bic

if __name__ == "__main__":
    time_dict = {2019: [8, 9]}
    data_type = "train"
    fill_method = "median"

    dataprovider = ModelDataProvider(time_dict=time_dict, data_type=data_type, fill_method=fill_method)
    feature_data = dataprovider.get_features_data()
    yield_data = dataprovider.get_yield_data()

    ols_1 = Ols_1(x=feature_data, y=yield_data)
    beta_1, x_1, y_preidct_1 = ols_1.cal_predict()
    ols_1.cal_R()

    ols_2 = Ols_2(x=feature_data, y=yield_data)
    beta_2, x_2, y_preidct_2 = ols_2.cal_predict()
    ols_2.cal_R()

    ols = Ols(x=feature_data, y=yield_data)
    y_preidct_ = ols.cal_predict()
    ols.cal_R()