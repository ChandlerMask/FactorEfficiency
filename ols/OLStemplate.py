import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy import linalg


#  definition of Ols
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
        L = linalg.cholesky(self.x_center)
        b, _ = linalg.lapack.dpotrs(L, self.xy_center)
        b0 = self.y_bar - np.inner(self.x_bar, b)
        beta = np.r_[b0, b]
        # print(L, "\n", self.xy_center, "\n", b, "\n", b0)
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



