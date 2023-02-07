"""Data exploration module
"""

import os

from loguru import logger
import pandas as pd

from excel.analysis.utils import statistics
from excel.analysis.utils import analyse_variables
from excel.analysis.utils import dim_reduction


class ExploreData:
    def __init__(
        self,
        data: pd.DataFrame,
        experiment: str,
        exploration: str,
        out_dir: str,
        metadata: list,
        remove_outliers: bool = False,
        investigate_outliers: bool = False,
        whis: float = 1.5,
        seed: int = 0,
        corr_thresh: float = 0.6,
        drop_features: bool = True,
        feature_reduction: str = 'forest',
    ) -> None:
        self.data = data
        self.experiment = experiment
        self.exploration = exploration
        self.out_dir = out_dir
        self.metadata = list(set(metadata) & set(data.columns))
        self.remove_outliers = remove_outliers
        self.investigate_outliers = investigate_outliers
        self.whis = whis
        self.seed = seed
        self.corr_thresh = corr_thresh
        self.drop_features = drop_features
        self.feature_reduction = feature_reduction

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
