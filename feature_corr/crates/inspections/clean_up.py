import os

import pandas as pd
from loguru import logger

from feature_corr.data_borg import DataBorg


class CleanUp(DataBorg):
    """ "Clean up data"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_as_index = self.config.inspection.label_as_index
        logger.info(f'Running -> {self.__class__.__name__}')
        self.clean_frame = pd.DataFrame()
        self.frame = self.get_frame('ephemeral')
        self.label_index_frame = None

    def __call__(self):
        self.label_index_frame = self.get_label_index_frame()
        self.get_consistent_frame()
        self.set_frame('ephemeral', self.clean_frame)
        if self.config.inspection.export_cleaned_frame:
            self.export_frame()

    def export_frame(self):
        """Export frame"""
        file_path = os.path.join(self.config.meta.output_dir, self.config.meta.name, 'cleaned_up_frame.xlsx')
        self.clean_frame.to_excel(file_path)
        logger.info(f'Exported cleaned frame to -> {file_path}')

    def get_label_index_frame(self):
        """Get label index frame"""
        if isinstance(self.label_as_index, str):
            if self.label_as_index not in self.frame:
                raise ValueError('Label index name is not in the data')
            self.label_index_frame = self.frame[self.label_as_index]

    def set_index_by_label(self):
        """Set index by label"""
        logger.info(f'Reindex table by name -> {self.label_as_index}')
        if isinstance(self.label_as_index, str):
            if self.label_as_index not in self.get_frame('ephemeral').columns:
                raise ValueError(f'Label {self.label_as_index} not in data')
            frame = self.get_frame('ephemeral')
            frame = frame.set_index(self.label_as_index)
            print(frame.head())
            self.set_frame('ephemeral', frame)

    def column_based_clean_up(self):
        """Clean up columns"""
        # TODO: Make me nice, always expect the unexpected

    def get_consistent_frame(self):
        """Get consistent frame"""
        for x_type in ['int64', 'float64']:
            pure_cols = self.frame.select_dtypes(include=x_type).columns
            self.clean_frame = pd.concat([self.clean_frame, self.frame[pure_cols]])
        self.clean_frame.dropna(how='all', axis=1, inplace=True)  # Drop columns with all NaN
        logger.info(f'Found data type consistent columns -> {len(self.clean_frame.columns)}/{len(self.frame.columns)}')

    def deal(self):
        """Set data type for columns"""
        logger.info(f'Column based data type')
        frame = self.get_frame('ephemeral')
