import pandas as pd
from sklearn import preprocessing


class Normaliser:
    def __init__(self) -> None:
        self.target_label = None

    def l1_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """L1 normalise data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.Normalizer(norm='l1').fit_transform(x.T).T  # l1 normalisation with transposed data
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data

    def l2_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """L2 normalise data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.Normalizer(norm='l2').fit_transform(x.T).T  # l2 normalisation with transposed data
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data

    def z_score_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z score data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.StandardScaler().fit_transform(x)
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data

    def min_max_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Min max scale data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.MinMaxScaler().fit_transform(x)  # default is 0-1
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data

    def max_abs_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Max abs scale data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.MaxAbsScaler().fit_transform(x)  # default is 0-1
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data

    def robust_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust scale data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.RobustScaler().fit_transform(x)
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data

    def quantile_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Quantile transform data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.QuantileTransformer().fit_transform(x)
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data

    def power_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Power transform data"""
        tmp = data[self.target_label]  # keep label col as is
        data = data.drop(self.target_label, axis=1)  # now remove column
        x = data.values  # returns a numpy array
        x_trans = preprocessing.PowerTransformer().fit_transform(x)
        data = pd.DataFrame(x_trans, columns=data.columns)
        data = data.join(tmp)
        return data
