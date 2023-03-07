import numpy as np
import pandas as pd
from loguru import logger
from sklearn.experimental import enable_iterative_imputer  # because of bug in sklearn
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer

logger.trace(enable_iterative_imputer)  # to avoid auto import removal
from cardio_parsers.data_borg import DataBorg


def data_bubble(func):
    def wrapper(self, *args):
        data = args[0]
        impute = func(self)
        imp_data = impute.fit_transform(data)
        imp_data = pd.DataFrame(imp_data, index=data.index, columns=data.columns)
        logger.info(f'{self.impute_method} reduced features from {len(data)} -> {len(imp_data)}')
        return imp_data

    return wrapper


class Imputers(DataBorg):
    """Impute missing data"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.seed = config.meta.seed
        self.impute_method = config.impute.method

    def __call__(self) -> None:
        """Impute missing data"""
        if self.__check_methods():
            return getattr(self, self.impute_method)(self.get_ephemeral_data())

    def __check_methods(self) -> bool:
        """Check if the given method is valid"""
        valid_methods = set([func for func in dir(self) if callable(getattr(self, func)) and not func.startswith('_')])
        method = set([self.impute_method])  # brackets to avoid splitting string into characters
        if not method.issubset(valid_methods):
            raise ValueError(f'Unknown imputation method: {self.impute_method}')
        return True

    def drop_nan_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop patients with any NaN values"""
        imp_data = data.dropna(axis=0, how='any')
        logger.info(f'{self.impute_method} reduced features from {len(data)} -> {len(imp_data)}')
        return imp_data

    @data_bubble
    def iterative_impute(self) -> IterativeImputer:
        """Iterative impute"""
        return IterativeImputer(
            initial_strategy='median',
            max_iter=100,
            random_state=self.seed,
            keep_empty_features=True,
        )

    @data_bubble
    def simple_impute(self) -> SimpleImputer:
        """Simple impute"""
        return SimpleImputer(
            strategy='median',
            keep_empty_features=True,
        )

    @data_bubble
    def missing_indicator_impute(self) -> MissingIndicator:
        """Missing indicator impute"""
        return MissingIndicator(
            missing_values=np.nan,
            features='all',
        )

    @data_bubble
    def knn_impute(self) -> KNNImputer:
        """KNN impute"""
        return KNNImputer(
            n_neighbors=5,
            keep_empty_features=True,
        )
