#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：dnnmodels.py
@Author  ：陈政霖
@Date    ：2023-04-08 21:42
"""

import pandas as pd
import numpy as np
import torch
from .dnntemplate import DeepLearningTemplate
from .loss import LossSharpe, LossRatio


class DeepLearningSharpe(DeepLearningTemplate):
    loss_class = LossSharpe

    def predict(self, x_predict: torch.tensor):
        if self.train_flag is False:
            self.training()
        y_predict = self.model(x_predict)
        return y_predict


class DeepLearningRatio(DeepLearningTemplate):
    loss_class = LossRatio

    def predict(self, x_predict: torch.tensor):
        if self.train_flag is False:
            self.training()
        y_predict = self.model(x_predict)
        return y_predict[:, 0]  # 返回正例（上涨的概率）