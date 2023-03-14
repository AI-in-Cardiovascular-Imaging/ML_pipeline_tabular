import os
import re

import numpy as np
import pandas as pd
from loguru import logger

from feature_corr.data_borg import DataBorg

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


class CleanUp(DataBorg):
    """ "Clean up data"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.label_as_index = self.config.inspection.label_as_index
        logger.info(f'Running -> {self.__class__.__name__}')
        self.auto_clean = self.config.inspection.auto_clean
        self.manual_clean = self.config.inspection.manual_clean
        self.clean_frame = None
        self.frame = self.get_frame('ephemeral')
        self.label_index_frame = None
        self.target_labels = self.config.meta.target_label

    def __call__(self) -> None:
        """Autoclean, manual clean or do nothing"""
        self.maybe_set_index_by_label()
        self.clean_frame = pd.DataFrame().reindex_like(self.frame)

        if self.auto_clean:
            self.get_consistent_data_types()

        if self.manual_clean:
            self.drop_columns_rex()

        if not self.auto_clean and not self.manual_clean:
            self.clean_frame = self.frame

        self.clean_frame = self.clean_frame.dropna(how='all', axis=1)  # Drop columns with all NaN
        self.clean_frame = self.clean_frame.replace(r'^\s*$', np.nan, regex=True)  # Replace empty strings with NaN

        self.set_frame('ephemeral', self.clean_frame)
        if self.config.inspection.export_cleaned_frame:
            self.export_frame()

    def export_frame(self) -> None:
        """Export frame"""
        file_path = os.path.join(self.config.meta.output_dir, self.config.meta.name, 'cleaned_up_frame.xlsx')
        self.clean_frame.to_excel(file_path)
        logger.info(f'Exported cleaned frame to -> {file_path}')

    def maybe_set_index_by_label(self) -> None:
        """Set index by label"""
        if isinstance(self.label_as_index, str):
            logger.info(f'Reindex table by name -> {self.label_as_index}')
            self.frame = self.frame.set_index(self.label_as_index)
            self.frame.sort_index(inplace=True)

    def get_consistent_data_types(self) -> None:
        """Get all columns with consistent data types"""
        frame = self.frame.replace(r'^\s*$', np.nan, regex=True)  # Replace empty strings with NaN
        for x_type in ['int64', 'float64']:
            pure_cols = list(frame.select_dtypes(include=x_type).columns)
            for col in pure_cols:
                self.clean_frame[col] = self.frame[col]
        clean_frame = self.clean_frame.dropna(how='all', axis=1)  # Drop columns with all NaN
        logger.info(f'Found data type consistent columns -> {len(clean_frame.columns)}/{len(self.frame.columns)}')

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
        for target_label in self.target_labels:
            y_frames = pd.concat([y_frames, self.clean_frame[target_label]], axis=1)
        x_frame = self.clean_frame.drop(self.target_labels, axis=1)
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
                logger.info(f'Dropped {diff_drop} columns by regex -> {drop_col_names}')
            self.clean_frame = pd.concat([x_frame, y_frames], axis=1)
