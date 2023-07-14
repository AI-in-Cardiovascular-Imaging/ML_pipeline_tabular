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
        tmp_label = frame[self.target_label]  # keep label col as is
        arr_frame = frame.values  # returns a numpy array
        norm_frame = func(self, arr_frame)
        norm_frame = pd.DataFrame(norm_frame, index=frame.index, columns=frame.columns)
        norm_frame[self.target_label] = tmp_label
        return norm_frame, None

    return wrapper


class Normalisers:
    """Normalise frame"""

    def __init__(self, target_label=None) -> None:
        self.target_label = target_label
        self.auto_norm_method = None
        self.scaler = None

    @data_bubble
    def l1_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """L1 normalise frame"""
        self.scaler = preprocessing.Normalizer(norm='l1')
        return self.scaler.fit_transform(frame.T).T  # l1 normalisation with transposed frame

    @data_bubble
    def l2_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        """L2 normalise frame"""
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

    def auto_norm(self, frame: pd.DataFrame) -> tuple:
        """Auto normalise frame based on data type per column"""
        if frame.isna().any(axis=None):
            raise ValueError('Data contains NaN values, consider imputing data')
        tmp_label = frame[self.target_label]  # keep label col as i
        for col_name in frame.columns:  # iterate over columns
            col_type = str(frame[col_name].dtype)
            col_unique_vals = frame[col_name].nunique()
            frame = self.__binary_norm(frame, col_name, col_type, col_unique_vals)
            frame = self.__continuous_norm(frame, col_name, col_type, col_unique_vals)
            frame = self.__object_norm(frame, col_name, col_type)
            frame = self.__datatime_norm(frame, col_name, col_type)
        frame[self.target_label] = tmp_label
        return frame, None

    def __normalise_accordingly(self, frame: pd.DataFrame, col_name: str, data_type_name: str) -> pd.DataFrame:
        """Normalise frame according to data type"""
        logger.trace(f'{data_type_name.capitalize()} data detected in {col_name}')
        col_data = frame[col_name]
        col_values = col_data.values.reshape(-1, 1)
        norm_method = self.auto_norm_method[data_type_name]
        norm = getattr(self, norm_method)
        ori_norm = norm.__wrapped__  # get original unwrapped function
        norm_col_values = ori_norm(self, col_values)
        col_data = pd.Series(norm_col_values.reshape(-1), index=col_data.index)
        frame[col_name] = col_data
        return frame

    def __binary_norm(self, frame: pd.DataFrame, col_name: str, col_type: str, col_unique_vals: int) -> pd.DataFrame:
        """Normalise binary data if present"""
        if ('int' in col_type or 'float' in col_type) and col_unique_vals == 2:
            frame = self.__normalise_accordingly(frame, col_name, data_type_name='binary')
        return frame

    def __continuous_norm(
        self, frame: pd.DataFrame, col_name: str, col_type: str, col_unique_vals: int
    ) -> pd.DataFrame:
        """Normalise continuous data if present"""
        if ('int' in col_type or 'float' in col_type) and col_unique_vals > 2:
            frame = self.__normalise_accordingly(frame, col_name, data_type_name='continuous')
        return frame

    def __object_norm(self, frame: pd.DataFrame, col_name: str, col_type: str) -> pd.DataFrame:
        """Normalise object data if present"""
        if 'object' in col_type:
            # frame = self.__normalise_accordingly(frame, col_name, data_type_name='object')
            logger.warning('Object normalisation is not implemented yet')
            raise NotImplementedError('Object normalisation is not implemented yet')
        return frame

    def __datatime_norm(self, frame: pd.DataFrame, col_name: str, col_type: str) -> pd.DataFrame:
        """Normalise datetime data if present"""
        if 'datetime' in col_type:
            # frame = self.__normalise_accordingly(frame, col_name, data_type_name='datetime')
            raise NotImplementedError('Datetime normalisation is not implemented yet')
        return frame
