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
        self.experiment_name = config.analysis.experiment.name
        self.impute = config.merge.impute
        self.overwrite = config.merge.overwrite
        self.update_metadata = config.merge.update_metadata

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

        # Update metadata if desired (only makes sense if overwrite=False)
        if not self.overwrite and self.update_metadata:
            updater = UpdateMetadata(data, self.config)
            data = updater()

        data = data.set_index('subject')  # Use subject ID as index column
        explorer = ExploreData(data, self.config)
        explorer()


if __name__ == '__main__':

    @hydra.main(version_base=None, config_path='../../config', config_name='config')
    def main(config: DictConfig) -> None:
        analysis = Analysis(config)
        analysis()

    main()
