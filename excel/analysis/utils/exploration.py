"""Data exploration module
"""

import os

from loguru import logger
import pandas as pd
from omegaconf import DictConfig

from excel.analysis.utils import analyse_variables
from excel.analysis.utils import dim_reduction
from excel.analysis.utils.helpers import normalise_data


class ExploreData:
    def __init__(self, data: pd.DataFrame, config: DictConfig) -> None:
        self.data = data
        self.out_dir = os.path.join(config.dataset.out_dir, '6_exploration', config.analysis.experiment.name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.target_label = config.analysis.experiment.target_label
        self.exploration = config.analysis.run.exploration
        self.remove_outliers = config.analysis.run.remove_outliers
        self.investigate_outliers = config.analysis.run.investigate_outliers
        self.metadata = config.analysis.experiment.metadata
        self.seed = config.analysis.run.seed
        self.feature_reduction = config.analysis.run.feature_reduction
        self.corr_thresh = config.analysis.run.corr_thresh
        self.drop_features = config.analysis.run.drop_features

    def __call__(self) -> None:
        if self.remove_outliers or self.investigate_outliers:  # detect (and optionally remove or investigate) outliers
            self.data = analyse_variables.detect_outliers(
                self.data,
                out_dir=self.out_dir,
                remove=self.remove_outliers,
                investigate=self.investigate_outliers,
                metadata=self.metadata,
            )

        if 'univariate_analysis' in self.exploration:  # perform univariate analysis before feature reduction
            analyse_variables.univariate_analysis(
                self.data,
                out_dir=self.out_dir,
                metadata=self.metadata,
                hue=self.target_label,
            )
            self.exploration.remove('univariate_analysis')

        if 'correlation' in self.exploration:  # analyse correlation between variables
            self.data, self.metadata = analyse_variables.correlation(
                self.data,
                self.out_dir,
                self.metadata,
                corr_thresh=self.corr_thresh,
                drop_features=self.drop_features,
            )
            self.exploration.remove('correlation')

        self.data = normalise_data(self.data, self.target_label)  # normalise data

        if self.feature_reduction is not None:
            logger.info(f'Performing {self.feature_reduction}-based feature reduction.')
            self.data, self.metadata = analyse_variables.feature_reduction(
                self.data,
                self.out_dir,
                self.metadata,
                method=self.feature_reduction,
                seed=self.seed,
                label=self.target_label,
            )
            logger.info(f'{self.feature_reduction}-based feature reduction finished.')

        for expl in self.exploration:
            logger.info(f'Performing {expl} data exploration for {len(self.data.index)} patients.')
            try:
                stats_func = getattr(analyse_variables, expl)
                stats_func(self.data, self.out_dir, self.metadata, self.target_label)
                logger.info(f'{expl} data exploration finished.')
                continue
            except AttributeError:
                pass

            try:
                stats_func = getattr(dim_reduction, expl)
                stats_func(self.data, self.out_dir, self.metadata, self.target_label, self.seed)
                logger.info(f'{expl} data exploration finished.')
                continue
            except AttributeError:
                raise NotImplementedError
