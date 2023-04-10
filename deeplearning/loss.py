#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FactorEfficiency 
@File    ：loss.py
@Author  ：陈政霖
@Date    ：2023-04-08 16:40 
'''

import torch
import pandas as pd
import numpy as np
from abc import abstractmethod


class LossTemplate(object):
    loss_name = None

    def __init__(self, y_hat: torch.tensor, y: torch.tensor, ratio: float):
        self.y_hat = y_hat
        self.y = y
        self.ratio = ratio
        pass

    def get_loss_name(self):
        print("the loss is about {}".format(self.loss_name))
        pass

    @abstractmethod
    def cal_loss(self):
        pass

# def _cal_withdraw(data: np.array, index: int):
#     cal_data = data[: index+1]
#     highest = cal_data.max()
#     return highest - cal_data[-1]
#
# def cal_withdraw(data: np.array):
#     result_data = [_cal_withdraw(data, index) for index in range(len(data))]
#     return max(result_data)


class LossSharpe(LossTemplate):
    """
    在本类中，输入的y为因子收益率矩阵（T*n)，y_hat为权重矩阵（n*1）
    返回的损失函数值为 负的夏普率
    """
    loss_name = "sharpe"

    def _pick_factors(self):
        """
        根据ratio挑选出交易标的
        """
        slice_point = np.quantile(self. y_hat, (1 - self.ratio))
        filted_grades = torch.where(self.y_hat > slice_point, self.y_hat, 0)
        return filted_grades

    def _get_weights(self):
        filted_grades = self._pick_factors()
        weights = filted_grades / filted_grades.sum()
        return weights

    def cal_loss(self):
        weights = self._get_weights()
        n = torch.count_nonzero(weights)  # 统计交易因子个数
        return_series = torch.mm(self.y, weights) / n
        t = self.y.shape[0]  # 交易天数
        std_ = return_series.std() * np.sqrt(252 / t)
        mean_ = return_series.sum() * (252 / t)
        return mean_ / std_


class LossRatio(LossTemplate):
    """
    输入y_hat:(n*2)：n个样本涨跌的概率；y：实际涨跌的01变量
    return: 所选交易标的 涨跌概率的交叉熵
    """
    loss_name = "ratio"

    def _get_filter_vector(self):
        n_positive, _ = torch.quantile(self.y_hat, q=1 - self.ratio, dim=0)
        tool_p = self.y_hat[:, 0]
        filter_p = torch.where((tool_p > n_positive), 1, 0)
        return filter_p

    def cal_loss(self):
        filter_matrix = self._get_filter_vector()
        log = torch.log(self.y_hat[range(len(self.y_hat)), self.y])
        loss = (log * filter_matrix).sum()
        return loss
