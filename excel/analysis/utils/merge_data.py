"""Extracts data for desired experiment
"""

import os

from loguru import logger
import pandas as pd
from omegaconf import DictConfig
from sklearn.experimental import enable_iterative_imputer  # because of bug in sklearn
from sklearn.impute import SimpleImputer, IterativeImputer

from excel.global_helpers import checked_dir
from excel.analysis.utils.helpers import save_tables


class MergeData:
    """Extracts data for given localities, dims, axes, orientations and metrics"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.src = config.dataset.out_dir
        dir_name = checked_dir(config.dataset.dims, config.dataset.strict)
        self.checked_src = os.path.join(config.dataset.out_dir, '4_checked', dir_name)
        self.dims = config.dataset.dims
        self.impute = config.merge.impute
        self.peak_values = config.merge.peak_values
        self.mdata_src = config.dataset.mdata_src
        self.target_label = config.analysis.experiment.target_label
        self.experiment_name = config.analysis.experiment.name
        self.axes = config.analysis.experiment.axes
        self.orientations = config.analysis.experiment.orientations
        self.metrics = config.analysis.experiment.metrics
        self.metadata = config.analysis.experiment.metadata
        self.seed = config.analysis.run.seed
        self.segments = config.analysis.experiment.segments

        if self.metadata:  # always want subject IDs and label
            to_add = ['redcap_id', 'pat_id', self.target_label]
            self.metadata.extend([col for col in to_add if col not in self.metadata])
        else:
            self.metadata = [self.target_label]

        self.relevant = []
        self.table_name = None

    def __call__(self) -> None:
        logger.info('Merging data according to config parameters...')
        tables_list = []
        self.identify_tables()  # identify relevant tables w.r.t. input parameters
        subjects = os.listdir(self.checked_src)
        for subject in subjects:  # loop over subjects
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
        tables = tables.rename_axis('subject').reset_index()  # add a subject column and reset index
        tables['subject'] = tables['subject'].astype(int)

        if self.impute:  # data imputation (merged data)
            imputed_tables = self.impute_data(tables)
            tables = pd.DataFrame(imputed_tables, index=tables.index, columns=tables.columns)
        else:  # remove patients with any NaN values
            tables = tables.dropna(axis=0, how='any')

        if self.metadata:  # read and clean metadata
            tables = self.add_metadata(tables)

        tables = tables.sort_values(by='subject')  # save the tables for analysis
        save_tables(src=self.src, experiment_name=self.experiment_name, tables=tables)

    def __del__(self) -> None:
        logger.info('Data merging finished.')

    def identify_tables(self) -> None:
        for segment in self.segments:
            for dim in self.dims:
                for axis in self.axes:
                    for orientation in self.orientations:
                        if (
                            axis == 'short_axis'
                            and orientation == 'longit'
                            or axis == 'long_axis'
                            and orientation == 'circumf'
                            or axis == 'long_axis'
                            and orientation == 'radial'
                        ):
                            continue  # skip impossible or imprecise combinations

                        for metric in self.metrics:
                            self.relevant.append(f'{segment}_{dim}_{axis}_{orientation}_{metric}')

    def loop_files(self, subject) -> pd.DataFrame:
        for root, _, files in os.walk(os.path.join(self.checked_src, subject)):
            files.sort()  # sort files for consistent order of cols among subjects
            for file in files:
                # consider only relevant tables
                for table_name in self.relevant:
                    if file.endswith('.xlsx') and f'{table_name}_(' in file:
                        # logger.info(f'Relevant table {table_name} found for subject {subject}.')
                        self.table_name = table_name
                        file_path = os.path.join(root, file)
                        table = pd.read_excel(file_path)
                        yield table

    def remove_time(self, table) -> pd.DataFrame:
        """Remove time columns from ROI analysis tables"""
        return table[table.columns.drop(list(table.filter(regex='time')))]

    def extract_peak_values(self, table) -> None:
        """Extract peak values from ROI analysis tables"""
        info_cols = 1 if 'aha' in self.table_name else 2  # AHA data got one info col, ROI data got two info cols

        if 'long_axis' in self.table_name:  # ensure consistent naming between short and long axis
            table = table.rename(columns={'series, slice': 'slice'})

        if 'roi' in self.table_name:  # ROI analysis, remove slice-wise global rows and  keep only global, endo, epi ROI
            table = table.drop(table[(table.roi == 'global') & (table.slice != 'all slices')].index)
            to_keep = ['global', 'endo', 'epi']
            table = table[table.roi.str.contains('|'.join(to_keep)) == True]

        if self.impute:  # data imputation (table-wise)
            table.iloc[:, info_cols:] = self.impute_data(table.iloc[:, info_cols:], categorical=[])
        # else: # remove patients with any NaN values
        #     table = table.dropna(axis=0, how='any')

        # Circumferential and longitudinal strain and strain rate peak at minimum value
        if 'strain' in self.table_name and ('circumf' in self.table_name or 'longit' in self.table_name):
            peak = table.iloc[:, info_cols:].min(axis=1, skipna=True)  # compute peak values over sample cols
        else:
            peak = table.iloc[:, info_cols:].max(axis=1, skipna=True)

        table = pd.concat([table.iloc[:, :info_cols], peak], axis=1)  # concat peak values to info cols

        if 'roi' in self.table_name:  # ROI analysis -> group by global/endo/epi
            table = table.groupby(by='roi', sort=False).agg('mean', numeric_only=True)  # remove slice-wise global rows

        col_names = []  # store column names for later
        for segment in to_keep:
            orientation = [o for o in self.orientations if o in self.table_name][0]
            metric = [m for m in self.metrics if m in self.table_name][0]
            col_names.append(f'{segment}_{orientation}_{metric}')
            self.col_names.append(f'{segment}_{orientation}_{metric}')

        self.subject_data = pd.concat((self.subject_data, pd.Series(list(table.iloc[:, 0]), index=col_names)), axis=0)

    def impute_data(self, table: pd.DataFrame, categorical: list = []):
        """Impute missing values in table"""
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

    def add_metadata(self, tables):
        """Add metadata to tables"""
        try:
            mdata = pd.read_excel(self.mdata_src)
        except FileNotFoundError:
            logger.error(f'Metadata file not found, check path: {self.mdata_src}' '\nContinue without metadata...')
            mdata = None

        if mdata is not None:
            mdata = mdata[self.metadata]
            # clean some errors in metadata
            if 'mace' in self.metadata:
                mdata[mdata['mace'] == 999] = 0
            if 'fhxcad___1' in self.metadata:
                mdata.loc[~mdata['fhxcad___1'].isin([0, 1]), 'fhxcad___1'] = 0

            # clean subject IDs
            mdata['pat_id'].fillna(mdata['redcap_id'], inplace=True)  # patients without pat_id get redcap_id
            mdata = mdata[mdata['pat_id'].notna()]  # remove rows without pat_id and redcap_id
            mdata = mdata.rename(columns={'pat_id': 'subject'})
            mdata['subject'] = mdata['subject'].astype(int)

            # merge the cvi42 data with available metadata
            tables = tables.merge(mdata, how='left', on='subject')
            tables = tables.drop('subject', axis=1)  # use redcap_id as subject id
            tables = tables[tables['redcap_id'].notna()]  # remove rows without redcap_id
            tables = tables.rename(columns={'redcap_id': 'subject'})

            # Impute missing metadata if desired
            if self.impute:
                categorical = [col for col in self.metadata if col not in ['bmi']]
                imputed_tables = self.impute_data(tables, categorical)
                tables = pd.DataFrame(imputed_tables, index=tables.index, columns=tables.columns)

                # convert integer cols explicitly to int
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

        return tables
