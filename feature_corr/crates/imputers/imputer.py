import numpy as np
import pandas as pd
from loguru import logger
from sklearn.experimental import enable_iterative_imputer  # because of bug in sklearn
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer

from feature_corr.data_borg import DataBorg

logger.trace(enable_iterative_imputer)  # to avoid auto import removal


def data_bubble(func):
    """Apply imputation method to frame"""

    def wrapper(self, *args) -> None:
        frame = args[0]
        impute = func(self)
        imp_frame = impute.fit_transform(frame)
        imp_frame = pd.DataFrame(imp_frame, index=frame.index, columns=frame.columns)
        if len(frame.columns) != len(imp_frame.columns):
            logger.info(
                f'{self.impute_method} reduced features from {len(frame.columns)-1} -> {len(imp_frame.columns)-1}'
            )
        self.set_store('frame', self.state_name, 'ephemeral', imp_frame)
        if len(imp_frame) == 0:
            raise ValueError('No cases left after dropping NaN values, use another imputation method or clean data')

    return wrapper


class Imputer(DataBorg):
    """Impute missing data"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.seed = config.data_split.seed
        self.state_name = config.meta.state_name
        self.impute_method = config.impute.method

    def __call__(self) -> None:
        """Impute missing data"""
        ephemeral_frame = self.get_store('frame', self.state_name, 'ephemeral')
        if self._check_methods():
            return getattr(self, self.impute_method)(ephemeral_frame)

    def verification_mode(self, frame: pd.DataFrame, seed: int) -> None:
        """Impute missing data for verification mode"""
        self.seed = seed
        self.state_name = 'verification'
        if self._check_methods():
            return getattr(self, self.impute_method)(frame)

    def _check_methods(self) -> bool:
        """Check if the given method is valid"""
        valid_methods = set([func for func in dir(self) if callable(getattr(self, func)) and not func.startswith('_')])
        method = set([self.impute_method])  # brackets to avoid splitting string into characters
        if not method.issubset(valid_methods):
            raise ValueError(f'Unknown imputation method -> "{self.impute_method}"')
        return True

    def drop_nan_impute(self, frame: pd.DataFrame) -> None:
        """Drop patients with any NaN values"""
        imp_frame = frame.dropna(axis=0, how='any')
        logger.info(f'{self.impute_method} reduced features from {len(frame)} -> {len(imp_frame)}')
        if len(imp_frame) == 0:
            raise ValueError('No cases left after dropping NaN values, use another imputation method or clean data')
        self.set_store('frame', self.state_name, 'ephemeral', imp_frame)

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
