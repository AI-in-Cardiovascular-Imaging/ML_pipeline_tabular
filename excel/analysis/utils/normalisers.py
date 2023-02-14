import pandas as pd
from sklearn import preprocessing


class Normaliser:

    def __init__(self) -> None:
        self.target_label = None

    def l1_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """L1 normalise data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        min_max_scaler = preprocessing.Normalizer(norm='l1')
        x_scaled = min_max_scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data

    def l2_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """L2 normalise data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        min_max_scaler = preprocessing.Normalizer(norm='l2')
        x_scaled = min_max_scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data

    def z_score_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z score data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data

    def min_max_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Min max scale data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()  # default is 0-1
        x_scaled = min_max_scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data

    def max_abs_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Max abs scale data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        scaler = preprocessing.MaxAbsScaler()  # default is 0-1
        x_scaled = scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data

    def robust_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust scale data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        scaler = preprocessing.RobustScaler()
        x_scaled = scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data

    def quantile_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Quantile transform data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        scaler = preprocessing.QuantileTransformer()
        x_scaled = scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data

    def power_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Power transform data"""
        tmp = data[self.target_label]  # keep label col as is
        x = data.values  # returns a numpy array
        scaler = preprocessing.PowerTransformer()
        x_scaled = scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        data[self.target_label] = tmp
        return data
