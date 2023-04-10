#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FactorEfficiency
@File    ：ratio_selected_test.py
@Author  ：陈政霖
@Date    ：2023-04-08 22:23
"""

import torch

ratio = 0.2

y = torch.tensor([1 for i in range(10)])
y_hat = torch.tensor([[0.1 * i, 1 - 0.1 * i] for i in range(10)], requires_grad=True)

n_positive, _ = torch.quantile(y_hat, q=1 - ratio, dim=0)
# n_negative = 1 - n_positive

tool_p = y_hat[:, 0]
# tool_n = y_hat[:, 1]

filter_p = torch.where((tool_p > n_positive), 1, 0)
# filter_n = torch.where((tool_n < n_negative), tool_n, 0).reshape(-1, 1)

# y_hat_filter = torch.cat([filter_p, filter_n], dim=1)

log = torch.log(y_hat[range(len(y_hat)), y])
loss = (log * filter_p).sum()


