"""Gets top features from all estimators and verifies does as well as the best estimator"""
"""Multiple runs with different seeds than specify"""
"""Exports summary of all found features per run and the best model"""

from loguru import logger

from feature_corr.data_borg import DataBorg


class Report(DataBorg):
    def __init__(self):
        super().__init__()
        self.features = None

    def __call__(self):
        logger.warning(f'Reporting started')
        self.features = self.get_all_features()
        for key, values in self.features.items():
            logger.warning(f'{key} -> {values}')
