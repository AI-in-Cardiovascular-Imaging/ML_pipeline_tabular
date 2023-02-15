import pandas as pd
from sklearn import preprocessing


def data_bubble(func):
    def wrapper(self, *args):
        data = args[0]
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        x_trans = func(self, x)
        x_trans = pd.DataFrame(x_trans, index=data.index, columns=data.columns)
        x_trans[self.target_label] = tmp
        return x_trans

    return wrapper


class Normaliser:
    def __init__(self) -> None:
        self.target_label = None

    @data_bubble
    def l1_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """L1 normalise data"""
        return preprocessing.Normalizer(norm='l1').fit_transform(data.T).T  # l1 normalisation with transposed data

    @data_bubble
    def l2_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """L2 normalise data"""
        return preprocessing.Normalizer(norm='l2').fit_transform(data.T).T  # l2 normalisation with transposed data

    @data_bubble
    def z_score_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z score data"""
        return preprocessing.StandardScaler().fit_transform(data)

    @data_bubble
    def min_max_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Min max scale data"""
        return preprocessing.MinMaxScaler().fit_transform(data)  # default is 0-1

    @data_bubble
    def max_abs_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Max abs scale data"""
        return preprocessing.MaxAbsScaler().fit_transform(data)  # default is 0-1

    @data_bubble
    def robust_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust scale data"""
        return preprocessing.RobustScaler().fit_transform(data)

    @data_bubble
    def quantile_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Quantile transform data"""
        return preprocessing.QuantileTransformer().fit_transform(data)

    @data_bubble
    def power_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Power transform data"""
        return preprocessing.PowerTransformer().fit_transform(data)
