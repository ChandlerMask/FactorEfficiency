import numpy as np
import itertools
import statsmodels.api as sm
from scipy.stats import norm
from scipy import linalg


# definition of data
class Data(object):
    """
    Data类：
    用于根据输入的参数随机生成模拟数据，并切割为X和y
    """

    def __init__(self, n: int, p: int, norm_dist=None, random_seed: int = None):
        np.random.seed(random_seed)

        if norm_dist is None:
            norm_dist = norm()

        self.x = norm_dist.rvs(size=(n, p))
        beta = np.array([1, -1] * int(p / 2))
        e = norm_dist.rvs(size=n)
        self.y = self.x.dot(beta) + e
        pass
