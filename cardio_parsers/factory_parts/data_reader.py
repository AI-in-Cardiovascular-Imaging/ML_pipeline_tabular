import os

import pandas as pd
from data_borg.data_borg import DataBorg
from loguru import logger


class DataReader(DataBorg):
    """Reads excel, csv, or pd dataframe and returns a pd dataframe"""

    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        if isinstance(self.path, str):
            if not os.path.isfile(self.path):
                raise FileNotFoundError(f'Invalid file path, check -> {self.path}')

    def __call__(self) -> None:
        """Reads excel, csv, or pd dataframe and returns a pd dataframe"""
        if self.path.endswith('.csv'):
            logger.info(f'Reading csv file -> {self.path}')
            self.set_original_data(pd.read_csv(self.path))
        elif self.path.endswith('.xlsx'):
            logger.info(f'Reading excel file -> {self.path}')
            self.set_original_data(pd.read_excel(self.path))
        elif isinstance(self.path, pd.DataFrame):
            logger.info(f'Reading dataframe -> {self.path}')
            self.set_original_data(self.path)
        else:
            raise ValueError(f'Found invalid file type, allowed is (.csv, .xlsx, dataframe), check -> {self.path}')
