import os

import pandas as pd
from loguru import logger

from feature_corr.data_borg import DataBorg


class CleanUp(DataBorg):
    """ "Clean up data"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.label_as_index = self.config.inspection.label_as_index
        logger.info(f'Running -> {self.__class__.__name__}')
        self.auto_clean = self.config.inspection.auto_clean
        self.manual_clean = self.config.inspection.manual_clean
        self.clean_frame = pd.DataFrame()
        self.frame = self.get_frame('ephemeral')
        self.label_index_frame = None

    def __call__(self) -> None:
        """Autoclean, manual clean or do nothing"""
        self.maybe_get_label_index_frame()

        if self.auto_clean:
            self.get_consistent_frame()

        if self.manual_clean:
            pass

        if not self.auto_clean and not self.manual_clean:
            self.clean_frame = self.frame

        self.maybe_set_index_by_label()

        self.set_frame('ephemeral', self.clean_frame)
        if self.config.inspection.export_cleaned_frame:
            self.export_frame()

    def export_frame(self) -> None:
        """Export frame"""
        file_path = os.path.join(self.config.meta.output_dir, self.config.meta.name, 'cleaned_up_frame.xlsx')
        self.clean_frame.to_excel(file_path)
        logger.info(f'Exported cleaned frame to -> {file_path}')

    def maybe_get_label_index_frame(self) -> None:
        """Get label index frame"""
        if isinstance(self.label_as_index, str):
            if self.label_as_index not in self.frame:
                raise ValueError(f'Label index name: "{self.label_as_index}" is not in the data')
            self.label_index_frame = self.frame[self.label_as_index]

    def maybe_set_index_by_label(self) -> None:
        """Set index by label"""
        if isinstance(self.label_as_index, str):
            logger.info(f'Reindex table by name -> {self.label_as_index}')
            tmp_frame = self.frame[[self.label_as_index]].copy()
            self.clean_frame = pd.concat([self.clean_frame, tmp_frame])
            self.clean_frame = self.clean_frame.set_index(self.label_as_index)

    def get_consistent_frame(self) -> None:
        """Get consistent frame"""
        for x_type in ['int64', 'float64']:
            pure_cols = self.frame.select_dtypes(include=x_type).columns
            self.clean_frame = pd.concat([self.clean_frame, self.frame[pure_cols]])
        self.clean_frame.dropna(how='all', axis=1, inplace=True)  # Drop columns with all NaN
        logger.info(f'Found data type consistent columns -> {len(self.clean_frame.columns)}/{len(self.frame.columns)}')

    def deal(self) -> None:
        """Set data type for columns"""
        logger.info(f'Column based data type')
        frame = self.get_frame('ephemeral')
