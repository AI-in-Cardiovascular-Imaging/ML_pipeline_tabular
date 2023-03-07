import os

import pandas as pd
from loguru import logger

from cardio_parsers.data_handler import DataHandler


class DataReader(DataHandler):
    """Reads excel, csv, or pd dataframe and returns a pd dataframe"""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.__dict__ = self._shared_state
        self.path = path
        if isinstance(self.path, str):
            if not os.path.isfile(self.path):
                raise FileNotFoundError(f'Invalid file path, check -> {self.path}')

    def __call__(self) -> None:
        """Reads excel, csv, or pd dataframe and returns a pd dataframe"""
        if self.path.endswith('.csv'):
            logger.info(f'Reading csv file -> {self.path}')
            self.store['original_frame'] = pd.read_csv(self.path)
        elif self.path.endswith('.xlsx'):
            logger.info(f'Reading excel file -> {self.path}')
            self.store['original_frame'] = pd.read_excel(self.path)
        elif isinstance(self.path, pd.DataFrame):
            logger.info(f'Reading dataframe -> {self.path}')
            self.store['original_frame'] = self.path
        else:
            raise ValueError(f'Found invalid file type, allowed is (.csv, .xlsx, dataframe), check -> {self.path}')