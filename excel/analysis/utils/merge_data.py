"""Extracts data for desired experiment
"""

import os

from loguru import logger
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from excel.global_helpers import checked_dir
from excel.analysis.utils.helpers import save_tables


class MergeData:
    """Extracts data for given localities, dims, axes, orientations and metrics"""

    def __init__(self, config: DictConfig) -> None:
        self.src = config.dataset.out_dir
        dir_name = checked_dir(config.dataset.dims)
        self.checked_src = os.path.join(self.src, '4_checked', dir_name)
        self.experiment = config.analysis.experiment
        self.dims = config.dataset.dims
        self.axes = config.analysis.axes
        self.orientations = config.analysis.orientations
        self.metrics = config.analysis.metrics
        self.impute = config.analysis.impute
        self.peak_values = config.analysis.peak_values
        self.metadata = config.analysis.metadata
        self.mdata_src = config.dataset.mdata_src
        self.seed = config.analysis.seed
        self.segments = config.analysis.segments

        # Always want subject ID
        if self.metadata:
            self.metadata = ['redcap_id', 'pat_id'] + self.metadata
        else:
            self.metadata = None

        self.relevant = []
        self.table_name = None

    def __call__(self) -> None:
        tables_list = []
        # Identify relevant tables w.r.t. input parameters
        self.identify_tables()
        # Parse source directory to read in relevant tables
        subjects = os.listdir(self.checked_src)
        for subject in subjects:
            self.col_names = []  # OPT: not necessary for each patient
            self.subject_data = pd.Series(dtype='float64')
            for table in self.loop_files(subject):
                if self.peak_values:
                    table = self.remove_time(table)
                    self.extract_peak_values(table)
                else:
                    logger.error('peak_values=False is not implemented yet.')
                    raise NotImplementedError
            tables_list.append(self.subject_data)

        # Build DataFrame from list (each row represents a subject)
        tables = pd.DataFrame(tables_list, index=subjects, columns=self.col_names)
        # Add a subject column and reset index
        tables = tables.rename_axis('subject').reset_index()
        tables['subject'] = tables['subject'].astype(int)

        # Data imputation (merged data)
        if self.impute:
            imputed_tables = self.impute_data(tables)
            tables = pd.DataFrame(imputed_tables, index=tables.index, columns=tables.columns)
        else:  # remove patients with any NaN values
            tables = tables.dropna(axis=0, how='any')

        # Read and clean metadata
        if self.metadata:
            try:
                mdata = pd.read_excel(self.mdata_src)
            except FileNotFoundError:
                logger.error(f'Metadata file not found, check path: {self.mdata_src}' '\nContinue without metadata...')
                mdata = None

            if mdata is not None:
                mdata = mdata[self.metadata]
                # Clean some errors in metadata
                if 'mace' in self.metadata:
                    mdata[mdata['mace'] == 999] = 0
                if 'fhxcad___1' in self.metadata:
                    mdata.loc[~mdata['fhxcad___1'].isin([0, 1]), 'fhxcad___1'] = 0

                # Clean subject IDs
                mdata['pat_id'].fillna(mdata['redcap_id'], inplace=True)  # patients without pat_id get redcap_id
                mdata = mdata[mdata['pat_id'].notna()]  # remove rows without pat_id and redcap_id
                mdata = mdata.rename(columns={'pat_id': 'subject'})
                mdata['subject'] = mdata['subject'].astype(int)

                # Merge the cvi42 data with available metadata
                tables = tables.merge(mdata, how='left', on='subject')
                tables = tables.drop('subject', axis=1)  # use redcap_id as subject id
                tables = tables[tables['redcap_id'].notna()]  # remove rows without redcap_id
                tables = tables.rename(columns={'redcap_id': 'subject'})

                # Remove any metadata columns containing less than thresh data
                threshold = 0.9
                tables = tables.dropna(axis=1, thresh=threshold * len(tables.index))

                # Remove these columns from the metadata list
                self.metadata = [col for col in self.metadata if col in tables.columns]

                # Impute missing metadata if desired
                if self.impute:
                    categorical = [col for col in self.metadata if col not in ['bmi']]
                    imputed_tables = self.impute_data(tables, categorical)
                    tables = pd.DataFrame(imputed_tables, index=tables.index, columns=tables.columns)

                    # Convert integer cols explicitly to int
                    int_cols = ['age']
                    for col in int_cols:
                        try:
                            tables[col] = tables[col].astype(int)
                        except KeyError:
                            pass  # skip if column is not found
                else:  # remove patients with any NaN values
                    logger.debug(f'Number of patients before dropping NaN metadata: {len(tables.index)}')
                    tables = tables.dropna(axis=0, how='any')
                    logger.debug(f'Number of patients after dropping NaN metadata: {len(tables.index)}')

                # Remove features containing the same value for all patients
                nunique = tables.nunique()
                cols_to_drop = nunique[nunique == 1].index
                tables = tables.drop(cols_to_drop, axis=1)

        # Save the tables for analysis
        tables = tables.sort_values(by='subject')
        save_tables(src=self.src, experiment=self.experiment, tables=tables)

    def identify_tables(self) -> None:
        for segment in self.segments:
            for dim in self.dims:
                for axis in self.axes:
                    for orientation in self.orientations:
                        # Skip impossible or imprecise combinations
                        if (
                            axis == 'short_axis'
                            and orientation == 'longit'
                            or axis == 'long_axis'
                            and orientation == 'circumf'
                            or axis == 'long_axis'
                            and orientation == 'radial'
                        ):
                            continue

                        for metric in self.metrics:
                            self.relevant.append(f'{segment}_{dim}_{axis}_{orientation}_{metric}')

    def loop_files(self, subject) -> pd.DataFrame:
        for root, _, files in os.walk(os.path.join(self.checked_src, subject)):
            files.sort()  # sort files for consistent order of cols among subjects
            for file in files:
                # Consider only relevant tables
                for table_name in self.relevant:
                    if file.endswith('.xlsx') and f'{table_name}_(' in file:
                        # logger.info(f'Relevant table {table_name} found for subject {subject}.')
                        self.table_name = table_name
                        file_path = os.path.join(root, file)
                        table = pd.read_excel(file_path)
                        yield table

    def remove_time(self, table) -> pd.DataFrame:
        return table[table.columns.drop(list(table.filter(regex='time')))]

    def extract_peak_values(self, table) -> None:
        # AHA data contain one info col, ROI data contains two info cols
        info_cols = 1 if 'aha' in self.table_name else 2

        # Ensure consistent naming between short and long axis
        if 'long_axis' in self.table_name:
            table = table.rename(columns={'series, slice': 'slice'})

        # ROI analysis
        if 'roi' in self.table_name:
            # Remove slice-wise global rows
            table = table.drop(table[(table.roi == 'global') & (table.slice != 'all slices')].index)
            # Keep only global, endo, epi ROI
            to_keep = ['global', 'endo', 'epi']
            table = table[table.roi.str.contains('|'.join(to_keep)) == True]

        # Data imputation (table-wise)
        if self.impute:
            table.iloc[:, info_cols:] = self.impute_data(table.iloc[:, info_cols:], categorical=[])
        # else: # remove patients with any NaN values
        #     table = table.dropna(axis=0, how='any')

        # Circumferential and longitudinal strain and strain rate peak at minimum value
        if 'strain' in self.table_name and ('circumf' in self.table_name or 'longit' in self.table_name):
            # Compute peak values over sample cols
            peak = table.iloc[:, info_cols:].min(axis=1, skipna=True)

        else:
            peak = table.iloc[:, info_cols:].max(axis=1, skipna=True)

        # Concat peak values to info cols
        table = pd.concat([table.iloc[:, :info_cols], peak], axis=1)

        # ROI analysis -> group by global/endo/epi
        if 'roi' in self.table_name:
            # Remove slice-wise global rows
            table = table.groupby(by='roi', sort=False).agg('mean', numeric_only=True)

        # Store column names for later
        col_names = []
        for segment in to_keep:
            orientation = [o for o in self.orientations if o in self.table_name][0]
            metric = [m for m in self.metrics if m in self.table_name][0]
            col_names.append(f'{segment}_{orientation}_{metric}')
            self.col_names.append(f'{segment}_{orientation}_{metric}')

        self.subject_data = pd.concat((self.subject_data, pd.Series(list(table.iloc[:, 0]), index=col_names)), axis=0)

    def impute_data(self, table: pd.DataFrame, categorical: list = []):
        cat_imputer = SimpleImputer(strategy='most_frequent')
        for col in categorical:
            try:
                table[col] = cat_imputer.fit_transform(table[[col]])
            except KeyError:
                pass  # skip if column is not found

        num_imputer = IterativeImputer(
            initial_strategy='median', max_iter=100, random_state=self.seed, keep_empty_features=True
        )
        # logger.debug(f'\n{table}')
        table = num_imputer.fit_transform(table)
        return table
