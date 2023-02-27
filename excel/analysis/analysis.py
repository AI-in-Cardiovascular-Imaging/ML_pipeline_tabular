""" Analysis module for all kinds of experiments
"""

import os
import sys

import hydra
from loguru import logger
from omegaconf import DictConfig
import pandas as pd

from excel.analysis.utils.merge_data import MergeData
from excel.analysis.utils.exploration import ExploreData
from excel.analysis.verifications import VerifyFeatures
from sklearn.model_selection import train_test_split

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)


class Analysis:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.src_dir = config.dataset.out_dir
        self.impute = config.merge.impute
        self.overwrite = config.merge.overwrite
        self.experiment_name = config.analysis.experiment.name
        self.target_label = config.analysis.experiment.target_label
        self.explore_frac = config.analysis.run.explore_frac

    def __call__(self) -> None:
        new_name = f'{self.experiment_name}_imputed' if self.impute else self.experiment_name
        merged_path = os.path.join(self.src_dir, '5_merged', f'{new_name}.xlsx')
        self.config.analysis.experiment.name = new_name

        # Data merging
        if os.path.isfile(merged_path) and not self.overwrite:
            logger.info('Merged data available, skipping merge step...')
        else:
            merger = MergeData(self.config)
            merger()

        data = pd.read_excel(merged_path)  # Read in merged data
        data = data.set_index('subject')  # Use subject ID as index column

        if 0 < self.explore_frac < 1:
            explore_data, verification_data = train_test_split(
                data, stratify=data[self.target_label], test_size=1-self.explore_frac
            )
        else:
            raise ValueError(f'Value {self.explore_frac} is invalid, must be float in (0, 1)')

        explorer = ExploreData(explore_data, self.config)
        features = explorer()

        verify = VerifyFeatures(self.config, verification_data, features)
        verify()


if __name__ == '__main__':

    @hydra.main(version_base=None, config_path='../../config', config_name='config')
    def main(config: DictConfig) -> None:
        logger.remove()
        logger.add(sys.stderr, level=config.logging_level)
        analysis = Analysis(config)
        analysis()

    main()
