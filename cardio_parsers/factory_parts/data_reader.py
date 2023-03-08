import os

import pandas as pd
from loguru import logger

from cardio_parsers.data_borg import DataBorg


class DataReader(DataBorg):
    """Reads excel, csv, or pd dataframe and returns a pd dataframe"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.state_name = config.meta.state_name
        self.file = config.meta.input_file
        if isinstance(self.file, str):
            if not os.path.isfile(self.file):
                raise FileNotFoundError(f'Invalid file path, check -> {self.file}')

    def __call__(self) -> None:
        self.add_state_name(self.state_name)
        if self._original_data is not None:
            self.copy_original_to_ephemeral(self.state_name)
        else:
            self.read_file()

    def read_file(self):
        """Reads excel, csv, or pd dataframe and returns a pd dataframe"""
        if self.file.endswith('.csv'):
            logger.info(f'Reading csv file -> {self.file}')
            data = pd.read_csv(self.file)
            self.set_original_data(data)
            self.set_ephemeral_data(self.state_name, data)

        elif self.file.endswith('.xlsx'):
            logger.info(f'Reading excel file -> {self.file}')
            data = pd.read_excel(self.file)
            self.set_original_data(data)
            self.set_ephemeral_data(self.state_name, data)

        elif isinstance(self.file, pd.DataFrame):
            logger.info(f'Reading dataframe -> {self.file}')
            self.set_original_data(self.file)
            self.set_ephemeral_data(self.state_name, self.file)

        else:
            raise ValueError(f'Found invalid file type, allowed is (.csv, .xlsx, dataframe), check -> {self.file}')
