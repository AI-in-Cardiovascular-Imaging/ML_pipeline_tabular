import os

import pandas as pd

from cardio_parsers.data_handler import DataHandler


class DataReader(DataHandler):
    """Reads excel, csv, or pd dataframe and returns a pd dataframe"""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.__dict__ = self._shared_state
        self.path = path
        if isinstance(self.path, str):
            if not os.path.isfile(self.path):
                raise ValueError(f'Invalid file path, check -> {self.path}')

    def __call__(self) -> None:
        """Reads excel, csv, or pd dataframe and returns a pd dataframe"""
        if self.path.endswith('.csv'):
            self.store['original_frame'] = pd.read_csv(self.path)
        if self.path.endswith('.xlsx'):
            self.store['original_frame'] = pd.read_excel(self.path)
        if isinstance(self.path, pd.DataFrame):
            self.store['original_frame'] = self.path
        raise ValueError(f'Invalid file type (.csv, .xlsx, dataframe), check -> {self.path}')
