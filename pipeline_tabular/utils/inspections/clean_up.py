import os
import re

import numpy as np
import pandas as pd
from loguru import logger

from pipeline_tabular.data_handler.data_handler import DataHandler


class CleanUp(DataHandler):
    """Clean up data"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.label_as_index = self.config.inspection.label_as_index
        self.manual_clean = self.config.inspection.manual_clean
        self.frame = self.get_frame()
        self.label_index_frame = None
        self.target_label = self.config.meta.target_label
        logger.info(f'Running -> {self.__class__.__name__}')

    def __call__(self) -> None:
        """Autoclean, manual clean or do nothing"""
        self.set_index_by_label()

        if self.manual_clean:
            self.drop_columns_rex()

        self.frame = self.frame.apply(pd.to_numeric, errors='coerce')  # Replace non-numeric entries with NaN
        nunique = self.frame.nunique()
        non_categorical = nunique[nunique > 5].index
        self.frame[non_categorical] = self.frame[non_categorical].replace(0, np.nan)  # Replace 0 with NaN
        self.frame = self.frame.dropna(how='all', axis=1)  # Drop columns with all NaN

        self.set_frame(self.frame)
        if self.config.inspection.export_cleaned_frame:
            self.export_frame()

    def export_frame(self) -> None:
        """Export frame"""
        output_dir = os.path.join(self.config.meta.output_dir, self.config.meta.name)
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, 'cleaned_up_frame.xlsx')
        self.frame.to_excel(file_path)
        logger.info(f'Exported cleaned frame to -> {file_path}')

    def set_index_by_label(self) -> None:
        """Set index by label"""
        if isinstance(self.label_as_index, str):
            logger.info(f'Reindex table by name -> {self.label_as_index}')
            self.frame = self.frame.set_index(self.label_as_index)
            # self.frame.sort_index(inplace=True)

    @staticmethod
    def _clean_up_regex(regex: str) -> list:
        """Clean up regex"""
        if regex is not None:
            if isinstance(regex, str):
                if regex.startswith('[') and regex.endswith(']'):
                    regex = regex[1:-1]
                if ',' in regex:
                    regex = regex.split(',')
                else:
                    regex = [regex]
            return regex

    def drop_columns_rex(self) -> None:
        """Drop columns by regex"""
        regex = self.config.inspection.manual_strategy.drop_columns_regex
        drop_regexes = self._clean_up_regex(regex)
        y_frames = pd.DataFrame()
        y_frames = pd.concat([y_frames, self.frame[self.target_label]], axis=1)
        x_frame = self.frame.drop(self.target_label, axis=1)
        if drop_regexes:
            for drop_regex in drop_regexes:
                expression = re.compile(r'{}'.format(drop_regex))
                tmp_frame = x_frame.copy()
                col_names = list(x_frame)

                drop_col_names = []
                for col_name in col_names:
                    if expression.search(col_name):
                        drop_col_names.append(col_name)

                x_frame = x_frame.drop(drop_col_names, axis=1)
                diff_drop = len(tmp_frame.columns) - len(x_frame.columns)
                logger.info(f'Dropped {diff_drop} columns by regex')
            self.frame = pd.concat([x_frame, y_frames], axis=1)
