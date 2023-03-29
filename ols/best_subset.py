import numpy as np
import itertools
from OLStemplate import Ols


class CvOls(object):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.shape[0] != y.shape[0]:
            raise ValueError("the x do not match y")
        self.x = x
        self.y = y

    def _cal_predict_ratio(self, x_train: np.ndarray, y_train: np.ndarray,
                           x_test: np.ndarray, y_test: np.ndarray):
        n = x_test.shape[0]
        ols = Ols(x=x_train, y=y_train)
        y_predict = ols.cal_predict(x=x_test)

        y_tilde = y_test - y_predict
        mse = y_tilde.dot(y_tilde)

        return mse / n

    def _k_fold(self, length: int, split: int, shuffle: bool = True, random_state=None):

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        index_list = list(range(length))

        if shuffle is True:
            random_state.shuffle(index_list)

        num = int(length / split)

        test_index_list = [index_list[i * num: (i + 1) * num] for i in range(split)]
        train_index_list = [(index_list[: i * num] + index_list[(i + 1) * num:]) for i in range(split)]

        return train_index_list, test_index_list

    def cal_predict_ratio_average(self, split: int = 5, shuffle: bool = True, random_state=None):
        if self.x.shape[0] <= split:
            raise ValueError("the x is to small")

        length = self.x.shape[0]
        train_index_list, test_index_list = self._k_fold(length=length, split=split,
                                                         shuffle=shuffle, random_state=random_state)

        x_train = [self.x[train_index_list[i]] for i in range(split)]
        y_train = [self.y[train_index_list[i]] for i in range(split)]
        x_test = [self.x[test_index_list[i]] for i in range(split)]
        y_test = [self.y[test_index_list[i]] for i in range(split)]

        mse_list = [self._cal_predict_ratio(x_train=x_train[i], y_train=y_train[i],
                                            x_test=x_test[i], y_test=y_test[i]) for i in range(split)]

        mse_average = sum(mse_list) / split
        return mse_average


class BestSubsetOls(object):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.result_dict = None
        pass

    def get_subset(self):
        num = self.x.shape[1]
        subset = []
        for i in range(1, num + 1):
            subset += list(itertools.combinations(list(range(num)), i))
        return subset

    def pick_variables(self, subset: set):
        x = self.x[:, subset]
        return x

    def cal_subsets(self, cv_split: int = 5, cv_shuffle: bool = True, cv_random_state=None):
        subset_list = self.get_subset()
        result_dict = {}

        models_list = [Ols(x=self.pick_variables(subset), y=self.y) for subset in subset_list]
        r_list = [model.cal_R() for model in models_list]
        aic_list = [model.cal_aic() for model in models_list]
        bic_list = [model.cal_bic() for model in models_list]

        cv_list = [CvOls(x=self.pick_variables(subset), y=self.y) for subset in subset_list]
        mse_list = [cv.cal_predict_ratio_average(split=cv_split, shuffle=cv_shuffle,
                                                 random_state=cv_random_state) for cv in cv_list]

        result_dict["subset"] = subset_list
        result_dict["models"] = models_list
        result_dict["r_squared"] = r_list
        result_dict["aic"] = aic_list
        result_dict["bic"] = bic_list
        result_dict["mse_average"] = mse_list

        return result_dict

    def get_indicators(self, cv_split: int = 5, cv_shuffle: bool = True, cv_random_state=None):
        """
        获取result_dict中的indicators指标
        """
        if self.result_dict is None:
            self.result_dict = self.cal_subsets(cv_split, cv_shuffle, cv_random_state)
        return [key for key in self.result_dict]

    def pick_best_subset(self, by: str):

        indicators_list = self.get_indicators()

        if by not in indicators_list:
            raise KeyError("所输入评价指标有误，请重新输入")
            pass

        indicator_list = self.result_dict[by]
        if by == "r_squared":
            best_indicator = max(indicator_list)
        else:
            best_indicator = min(indicator_list)

        print("the best {} is:".format(by), best_indicator)

        best_index = indicator_list.index(best_indicator)
        best_subset = self.result_dict["subset"][best_index]
        print("the best subset is:", best_subset)
        best_model = self.result_dict["models"][best_index]

        return best_subset, best_model