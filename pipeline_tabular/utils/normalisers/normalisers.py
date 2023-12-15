from functools import wraps

import pandas as pd
from loguru import logger
from sklearn import preprocessing


def data_bubble(func):
    """Pre and post processing for normalisation methods"""

    @wraps(func)
    def wrapper(self, *args):
        frame = args[0]
        if frame.isna().any(axis=None):
            raise ValueError('Data contains NaN values, consider imputing data')
        nunique = frame.nunique()
        non_categorical = list(nunique[nunique > 3].index)
        to_normalise = frame[non_categorical]
        tmp_label = frame[self.target_label]  # keep label col as is
        try:
            arr_frame = to_normalise.drop(self.target_label, axis=1)
        except KeyError:  # target label is categorical -> already removed
            arr_frame = to_normalise
        norm_frame = func(self, arr_frame)
        frame[non_categorical] = norm_frame
        frame[self.target_label] = tmp_label
        return frame, None

    return wrapper


class Normalisers:
    """Normalise frame"""

    def __init__(self, target_label=None) -> None:
        self.target_label = target_label
        self.scaler = None

    @data_bubble
    def l1_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """L1 normalise frame"""
        logger.warning('check implementation')
        self.scaler = preprocessing.Normalizer(norm='l1')
        return self.scaler.fit_transform(frame.T).T  # l1 normalisation with transposed frame

    @data_bubble
    def l2_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """L2 normalise frame"""
        logger.warning('check implementation')
        self.scaler = preprocessing.Normalizer(norm='l2')
        return self.scaler.fit_transform(frame.T).T  # l2 normalisation with transposed frame

    @data_bubble
    def z_score_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Z score frame"""
        self.scaler = preprocessing.StandardScaler()
        return self.scaler.fit_transform(frame)

    @data_bubble
    def min_max_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Min max scale frame"""
        self.scaler = preprocessing.MinMaxScaler()
        return self.scaler.fit_transform(frame)  # default is 0-1

    @data_bubble
    def max_abs_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Max abs scale frame"""
        self.scaler = preprocessing.MaxAbsScaler()
        return self.scaler.fit_transform(frame)  # default is 0-1

    @data_bubble
    def robust_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Robust scale frame"""
        self.scaler = preprocessing.RobustScaler()
        return self.scaler.fit_transform(frame)

    @data_bubble
    def quantile_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Quantile transform frame"""
        self.scaler = preprocessing.QuantileTransformer()
        return self.scaler.fit_transform(frame)

    @data_bubble
    def power_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Power transform frame"""
        self.scaler = preprocessing.PowerTransformer()
        return self.scaler.fit_transform(frame)
