""" Analysis module for all kinds of experiments
"""

import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import pandas as pd

from excel.analysis.utils.merge_data import MergeData
from excel.analysis.utils.update_metadata import UpdateMetadata
from excel.analysis.utils.exploration import ExploreData

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class Analysis:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.src_dir = config.dataset.out_dir
        self.experiment = config.analysis.experiment
        self.impute = config.analysis.impute
        self.overwrite = config.analysis.overwrite
        self.update_metadata = config.analysis.update_metadata
        self.exploration = config.analysis.exploration

    def __call__(self) -> None:
        # TODO: train and test paths/sets
        self.config.analysis.experiment = f'{self.experiment}_imputed' if self.impute else self.experiment
        merged_path = os.path.join(self.src_dir, '5_merged', f'{self.config.analysis.experiment}.xlsx')

        # Data merging
        if os.path.isfile(merged_path) and not self.overwrite:
            logger.info('Merged data available, skipping merge step...')
        else:
            logger.info('Merging data according to config parameters...')
            merger = MergeData(self.config)
            merger()
            logger.info('Data merging finished.')

        data = pd.read_excel(merged_path)  # Read in merged data

        # Update metadata if desired (only makes sense if overwrite=False)
        if not self.overwrite and self.update_metadata:
            logger.info('Updating metadata as requested...')
            updater = UpdateMetadata(data, self.config)
            data = updater()
            logger.info('Metadata update finished.')

        data = data.set_index('subject')  # Use subject ID as index column

        # Data exploration
        if self.exploration:
            expl_dir = os.path.join(self.src_dir, '6_exploration')
            os.makedirs(expl_dir, exist_ok=True)
            self.config.dataset.out_dir = expl_dir
            explorer = ExploreData(data, self.config)
            explorer()


if __name__ == '__main__':

    @hydra.main(version_base=None, config_path='../../config', config_name='config')
    def main(config: DictConfig) -> None:
        analysis = Analysis(config)
        analysis()

    main()
