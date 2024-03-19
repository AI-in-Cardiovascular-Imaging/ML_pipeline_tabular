import os

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.experimental import enable_iterative_imputer  # because of bug in sklearn
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer

from pipeline_tabular.data_handler.data_handler import DataHandler

logger.trace(enable_iterative_imputer)  # to avoid auto import removal

class Imputer(DataHandler):
    """Impute missing data"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.impute_method = config.impute.method
        self.imputer = None

    def __call__(self, seed) -> None:
        """Impute missing data"""
        self.seed = seed
        
        train_frame = self.get_store('frame', seed, 'train')
        test_frame = self.get_store('frame', seed, 'test')
        if self._check_methods():
            if self.impute_method == 'drop_nan_impute':
                raise NotImplementedError
            else:
                imputer = getattr(self, self.impute_method)()
                imp_train = imputer.fit_transform(train_frame)
                imp_train = pd.DataFrame(imp_train, index=train_frame.index, columns=train_frame.columns)
                imp_test = imputer.transform(test_frame)
                imp_test = pd.DataFrame(imp_test, index=test_frame.index, columns=test_frame.columns)
            self.set_store('frame', seed, 'train', imp_train)
            self.set_store('frame', seed, 'test', imp_test)
            if True:
                out_path = f'{os.path.splitext(self.config.meta.input_file)[0]}_imputed_{seed}'
                imp_train.to_csv(f'{out_path}_train.csv')
                imp_test.to_csv(f'{out_path}_test.csv')

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

    def iterative_impute(self) -> IterativeImputer:
        """Iterative impute"""
        return IterativeImputer(
            initial_strategy='median',
            max_iter=100,
            random_state=self.seed,
            keep_empty_features=True,
        )

    def simple_impute(self) -> SimpleImputer:
        """Simple impute"""
        return SimpleImputer(
            strategy='median',
            keep_empty_features=True,
        )

    def missing_indicator_impute(self) -> MissingIndicator:
        """Missing indicator impute"""
        return MissingIndicator(
            missing_values=np.nan,
            features='all',
        )

    def knn_impute(self) -> KNNImputer:
        """KNN impute"""
        return KNNImputer(
            n_neighbors=5,
            keep_empty_features=True,
        )
