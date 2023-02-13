"""Update metadata without re-running the entire merging process (e.g. peak calculations)
"""

import pandas as pd
from omegaconf import DictConfig
from loguru import logger

from excel.analysis.utils.helpers import merge_metadata, save_tables


class UpdateMetadata:
    def __init__(self, data: pd.DataFrame, config: DictConfig) -> None:
        self.data = data
        self.config = config
        self.src = config.dataset.out_dir
        self.experiment = config.analysis.experiment.name
        self.mdata_src = config.dataset.mdata_src
        self.metadata = config.analysis.experiment.metadata

    def __call__(self) -> pd.DataFrame:
        logger.info('Updating metadata as requested...')
        if self.experiment == 'layer_analysis':
            self.data = self.data.iloc[:, :10]  # Keep the first 10 cols, the rest are old metadata
        self.data = merge_metadata(self.data, self.mdata_src, self.metadata)  # Merge cvi42 data with new metadata
        save_tables(self.src, self.experiment, self.data)
        return self.data

    def __del__(self) -> None:
        logger.info('Metadata update finished.')
