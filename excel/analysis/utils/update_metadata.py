"""Update metadata without re-running the entire merging process (e.g. peak calculations)
"""

import pandas as pd
from omegaconf import DictConfig

from excel.analysis.utils.helpers import merge_metadata, save_tables


class UpdateMetadata:
    def __init__(self, data: pd.DataFrame, config: DictConfig) -> None:
        self.data = data
        self.config = config
        self.src = config.dataset.out_dir
        self.experiment = config.experiment.name
        self.mdata_src = config.dataset.mdata
        self.metadata = config.experiment.metadata

    def __call__(self) -> pd.DataFrame:
        if self.experiment == 'layer_analysis':
            # Keep the first 10 cols, the rest are old metadata
            self.data = self.data.iloc[:, :10]
        # Merge the cvi42 data with the new metadata
        self.data = merge_metadata(self.data, self.mdata_src, self.metadata)
        # Save the new data table for analysis
        save_tables(self.src, self.experiment, self.data)
        return self.data
