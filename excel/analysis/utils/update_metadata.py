"""Update metadata without re-running the entire merging process (e.g. peak calculations)
"""

import os

from loguru import logger
import pandas as pd

from excel.analysis.utils.helpers import merge_metadata, save_tables


class UpdateMetadata:
    def __init__(self, src: str, data: pd.DataFrame, mdata_src: str, metadata: list, experiment: str) -> None:
        self.src = src
        self.data = data
        self.mdata_src = mdata_src
        self.metadata = metadata
        self.experiment = experiment

    def __call__(self) -> pd.DataFrame:
        if self.experiment == 'layer_analysis':
            # Keep the first 10 cols, the rest are old metadata
            self.data = self.data.iloc[:, :10]
        # Merge the cvi42 data with the new metadata
        self.data = merge_metadata(self.data, self.mdata_src, self.metadata)

        # Save the new data table for analysis
        save_tables(self.src, self.experiment, self.data)

        return self.data
