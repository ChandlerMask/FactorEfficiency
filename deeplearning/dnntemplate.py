#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：dnntemplate.py
@Author  ：陈政霖
@Date    ：2023-04-07 11:07
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from .loss import LossSharpe
from abc import abstractmethod


class DeepLearningTemplate(object):

    loss_class = None
    train_flag = False

    def __init__(self, model: torch.nn.modules.container.Sequential, x: np.array, y: np.array, lr: float,
                 batch_size: int, epochs: int, ratio: float):
        self.model = model
        self.x = torch.tensor(x)
        self.y = torch.tensor(y).reshape(x.shape[0], -1)
        self.x_loader = DataLoader(dataset=self.x, batch_size=batch_size)
        self.y_loader = DataLoader(dataset=self.y, batch_size=batch_size)
        self.updater = torch.optim.SGD(model.parameters(), lr=lr)
        self.epochs = epochs
        self.ratio = ratio
        pass

    def _model_init(self, model):
        if isinstance(model, torch.nn.Linear):
            torch.nn.init.xavier_normal(model.weights)
            torch.nn.init.constant_(model.bias, 0)
        pass

    def model_init(self):
        self.model.apply(self._model_init())
        pass

    def _training(self):

        for x, y in self.x_loader, self.y_loader:
            self.updater.zero_grad()
            model_result = self.model(x)
            loss = self.loss_class.cal_loss(y_hat=model_result, y=self.y, ratio=self.ratio)
            loss.backward()
            self.updater.step()
        pass

    def training(self):
        for epoch in self.epochs:
            self._training()
            print("the {} time of training has finished".format(epoch))
        self.train_flag = True
        pass

    @abstractmethod
    def predict(self):
        """
        定义具体的training函数，在循环中执行_training以后，将模型输出进行转换：
        sharpe模型：直接输出权重.
        正收益模型：收入正收益概率
        """
        pass





