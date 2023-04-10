#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：evaluatortemplate.py
@Author  ：陈政霖
@Date    ：2023-04-08 18:36
"""

import torch
import numpy as np


class Evaluator(object):
    def __init__(self, grades: torch.tensor, y_summary: torch.tensor, ratio: float):
        """
        grades：模型输出的得分
        y_summary: summary收益率数据
        ratio：开仓比例
        """
        self.grades = grades
        self.y = y_summary
        self.ratio = ratio
        pass

    def _pick_factors(self):
        """
        根据ratio挑选出交易标的
        """
        slice_point = np.quantile(self.grades, (1 - self.ratio))
        filted_grades = torch.where(self.grades > slice_point, self.grades, 0)
        return filted_grades

    def _get_weights(self):
        filted_grades = self._pick_factors()
        weights = filted_grades / filted_grades.sum()
        return weights

    def cal_sharpe(self):
        weights = self._get_weights()
        n = torch.count_nonzero(weights)  # 统计交易因子个数
        return_series = torch.mm(self.y, weights) / n
        t = self.y.shape[0]  # 交易天数
        std_ = return_series.std() * np.sqrt(252 / t)
        mean_ = return_series.sum() * (252 / t)
        return mean_ / std_

    def cal_positive_ratio(self):
        weights = self._get_weights()
        n = torch.count_nonzero(weights)  # 统计交易因子个数
        weights = weights.reshape(len(weights), 1)  # 转为行向量
        return_matrix = self.y * weights
        return_total = return_matrix.sum(axis=1)  # 统计因子在窗口期内的总收益
        n_positive = torch.where(return_total > 0, 1, 0).sum()
        return n_positive / n
