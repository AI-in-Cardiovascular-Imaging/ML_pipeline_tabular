"""Data exploration module
"""

import os

from loguru import logger
import pandas as pd
from omegaconf import DictConfig

from excel.analysis.utils import statistics
from excel.analysis.utils import analyse_variables
from excel.analysis.utils import dim_reduction


class ExploreData:
    def __init__(self, data: pd.DataFrame, config: DictConfig) -> None:
        self.data = data
        self.out_dir = config.dataset.out_dir
        self.exploration = config.analysis.exploration
        self.remove_outliers = config.analysis.remove_outliers
        self.investigate_outliers = config.analysis.investigate_outliers
        self.whis = config.analysis.whis
        self.metadata = config.analysis.metadata
        self.seed = config.analysis.seed
        self.feature_reduction = config.analysis.feature_reduction
        self.corr_thresh = config.analysis.corr_thresh

    def __call__(self) -> None:
        # Detect (and optionally remove or investigate) outliers
        if self.remove_outliers or self.investigate_outliers:
            self.data = analyse_variables.detect_outliers(
                self.data,
                out_dir=self.out_dir,
                remove=self.remove_outliers,
                investigate=self.investigate_outliers,
                whis=self.whis,
                metadata=self.metadata,
            )

        if self.feature_reduction is not None:
            logger.info(f'Performing {self.feature_reduction}-based feature reduction.')
            self.drop_features = False  # higher precedence than correlation-based feature reduction
            self.data, self.metadata = analyse_variables.feature_reduction(
                self.data, self.out_dir, self.metadata, method=self.feature_reduction, seed=self.seed, label='mace'
            )
            logger.info(f'{self.feature_reduction}-based feature reduction finished.')

        for expl in self.exploration:
            logger.info(f'Performing {expl} data exploration for {len(self.data.index)} patients.')

            if expl == 'correlation':
                self.data, self.metadata = analyse_variables.correlation(
                    self.data,
                    self.out_dir,
                    self.metadata,
                    corr_thresh=self.corr_thresh,
                    drop_features=self.drop_features,
                )
                logger.info(f'{expl} data exploration finished.')
                continue

            try:
                stats_func = getattr(analyse_variables, expl)
                stats_func(self.data, self.out_dir, self.metadata, 'mace', self.whis)
                logger.info(f'{expl} data exploration finished.')
                continue
            except AttributeError:
                pass

            try:
                stats_func = getattr(dim_reduction, expl)
                stats_func(self.data, self.out_dir, self.metadata, 'mace', self.seed)
                logger.info(f'{expl} data exploration finished.')
                continue
            except AttributeError:
                raise NotImplementedError
