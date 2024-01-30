import os

import pandas as pd
from loguru import logger

from pipeline_tabular.data_handler.data_handler import DataHandler


class DataReader(DataHandler):
    """Reads excel, csv, or dataframe and returns a dataframe"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.file = config.meta.input_file
        if isinstance(self.file, str):
            if not os.path.isfile(self.file):
                raise FileNotFoundError(f'Invalid file path, check -> {self.file}')

    def __call__(self) -> None:
        self.read_file()
        self.set_frame(self.frame)

    def read_file(self):
        """Reads excel, csv, or pd dataframe and returns a pd dataframe"""
        if self.file.endswith('.csv'):
            logger.info(f'Reading csv file -> {self.file}')
            self.frame = pd.read_csv(self.file)

        elif self.file.endswith('.xlsx'):
            logger.info(f'Reading excel file -> {self.file}')
            self.frame = pd.read_excel(self.file)

        elif isinstance(self.file, pd.DataFrame):
            logger.info(f'Reading dataframe -> {self.file}')
            self.frame = self.file

        else:
            raise ValueError(f'Found invalid file type, allowed is (.csv, .xlsx, dataframe), check -> {self.file}')
