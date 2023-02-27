from functools import wraps

import pandas as pd
from loguru import logger
from sklearn import preprocessing


def data_bubble(func):
    @wraps(func)
    def wrapper(self, *args):
        data = args[0]
        if data.isna().any(axis=None):
            raise ValueError('Data contains NaN values, consider imputing data')
        tmp_label = data[self.target_label]  # keep label col as is
        arr_data = data.values  # returns a numpy array
        norm_data = func(self, arr_data)
        norm_data = pd.DataFrame(norm_data, index=data.index, columns=data.columns)
        norm_data[self.target_label] = tmp_label
        return norm_data

    return wrapper


class Normaliser:
    def __init__(self, target_label=None) -> None:
        self.target_label = target_label

        self.auto_norm_method = dict(
            {'binary': 'l1_norm', 'continuous': 'l2_norm', 'object': 'z_score_norm', 'datatime': 'min_max_norm'}
        )
        self.__check_auto_norm_keys()

    def __check_auto_norm_keys(self) -> None:
        """Check if auto_norm_method keys are valid"""
        valid_methods = set([x for x in dir(self) if not x.startswith('_') and x != 'process_job'])
        selected_methods = set(self.auto_norm_method.values())
        if not selected_methods.issubset(valid_methods):
            raise ValueError(f'Invalide job, check -> {str(selected_methods - valid_methods)}')

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
    def min_max_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Min max scale data"""
        return preprocessing.MinMaxScaler().fit_transform(data)  # default is 0-1

    @data_bubble
    def max_abs_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Max abs scale data"""
        return preprocessing.MaxAbsScaler().fit_transform(data)  # default is 0-1

    @data_bubble
    def robust_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust scale data"""
        return preprocessing.RobustScaler().fit_transform(data)

    @data_bubble
    def quantile_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Quantile transform data"""
        return preprocessing.QuantileTransformer().fit_transform(data)

    @data_bubble
    def power_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Power transform data"""
        return preprocessing.PowerTransformer().fit_transform(data)

    def auto_norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Auto normalise data based on data type per column"""
        if data.isna().any(axis=None):
            raise ValueError('Data contains NaN values, consider imputing data')
        tmp_label = data[self.target_label]  # keep label col as i
        for col_name in data.columns:  # iterate over columns
            col_type = str(data[col_name].dtype)
            col_unique_vals = data[col_name].unique()
            data = self.__binary_norm(data, col_name, col_type, col_unique_vals)
            data = self.__continuous_norm(data, col_name, col_type, col_unique_vals)
            data = self.__object_norm(data, col_name, col_type)
            data = self.__datatime_norm(data, col_name, col_type)
        data[self.target_label] = tmp_label
        return data

    def __normalise_accordingly(self, data, col_name, data_type_name) -> pd.DataFrame:
        logger.trace(f'{data_type_name.capitalize()} data detected in {col_name}')
        col_data = data[col_name]
        col_values = col_data.values.reshape(-1, 1)
        norm = getattr(self, self.auto_norm_method[data_type_name])
        ori_norm = norm.__wrapped__  # get original unwrapped function
        norm_col_values = ori_norm(self, col_values)
        col_data = pd.Series(norm_col_values.reshape(-1), index=col_data.index)
        data[col_name] = col_data
        return data

    def __binary_norm(self, data, col_name, col_type, col_unique_vals) -> pd.DataFrame:
        """Normalise binary data if present"""
        if ('int' in col_type or 'float' in col_type) and col_unique_vals.size == 2:
            data = self.__normalise_accordingly(data, col_name, data_type_name='binary')
        return data

    def __continuous_norm(self, data, col_name, col_type, col_unique_vals) -> pd.DataFrame:
        """Normalise continuous data if present"""
        if ('int' in col_type or 'float' in col_type) and col_unique_vals.size > 2:
            data = self.__normalise_accordingly(data, col_name, data_type_name='continuous')
        return data

    def __object_norm(self, data, col_name, col_type) -> pd.DataFrame:
        """Normalise object data if present"""
        if 'object' in col_type:
            # data = self.__normalise_accordingly(data, col_name, data_type_name='object')
            logger.warning('Object normalisation is not implemented yet')
        return data

    def __datatime_norm(self, data, col_name, col_type) -> pd.DataFrame:
        """Normalise datetime data if present"""
        if 'datetime' in col_type:
            # data = self.__normalise_accordingly(data, col_name, data_type_name='datetime')
            logger.warning('Datetime normalisation is not implemented yet')
        return data


if __name__ == '__main__':
    import datetime

    df = pd.DataFrame(
        {
            'A': [-1.12, -1.23, 2.32],
            'B': [1, 0, 1],
            'C': [1.0, 0.0, 1.0],
            'D': [1, None, 9],
            'E': [None, None, None],
            'F': ['A', 2, 3],
            'G': ['0', '1', '2'],
            'H': [pd.Timestamp('20180310'), '2009-01-06 15:08:24.789150', datetime.datetime.now()],
            'Z': ['A', 'V', 'C'],
        }
    )

    n = Normaliser('Z')
    a = n.auto_norm(df)
    print(a)
