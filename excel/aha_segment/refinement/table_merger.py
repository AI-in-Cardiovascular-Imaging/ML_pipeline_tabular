import os

import pandas as pd
from loguru import logger


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class MergeSegments:
    """Merge table of subjects"""

    def __init__(self, src: str, dst: str) -> None:
        self.src = src
        self.dst = dst
        self.memory = {}

    def __call__(self, dim, name) -> None:
        self.aggregate_data_frames(dim, name)
        self.merge_column_wise(dim, name)
        self.merge_row_wise(dim, name)

    def aggregate_data_frames(self, dim: str, name: str) -> None:
        """Aggregate data frames"""
        self.memory = {}
        for root, _, files in os.walk(self.src):
            if root.endswith(dim):  # filter w.r.t. dim
                for file in files:
                    if file.endswith('.xlsx') and name in file:
                        file_path = os.path.join(root, file)
                        logger.info(f'-> {file}')
                        df = pd.read_excel(file_path)
                        table_name = file.replace('.xlsx', '')
                        self.memory[table_name] = df

    def merge_column_wise(self, dim: str, table_name) -> None:
        """Merge columns of data frames"""
        columns = self.memory[list(self.memory.keys())[0]].columns  # get column names of first subject
        # cols here = [sample_0, sample_1, ...]
        for column in columns:
            if not 'AHA Segment' in column:
                df = pd.DataFrame(columns=self.memory.keys())
                for subject in self.memory:
                    x = self.memory[subject]
                    df[subject] = x[column]

                header = df.columns.tolist()
                header = [f'case_{x.split("_")[0]}' for x in header]
                df.columns = header
                df.rename(index={0: 'global'}, inplace=True)
                self.save(df, dim, f'aha_{dim}_{table_name}_{column}')

    def merge_row_wise(self, dim: str, table_name) -> None:
        """Merge columns of data frames"""
        # print(self.memory[list(self.memory.keys())[0]])
        tmp_df = self.memory[list(self.memory.keys())[0]]
        tmp_df = tmp_df.transpose()
        columns = tmp_df.columns.tolist()
        # cols here = [0, 1, ...]
        for column in columns:
            df = pd.DataFrame(columns=self.memory.keys())

            for subject in self.memory:
                x = self.memory[subject].transpose()
                # print(x)
                # print(column)
                df[subject] = x[column]

            header = df.columns.tolist()
            header = [f'case_{x.split("_")[0]}' for x in header]
            df.columns = header
            df.rename(index={0: 'global'}, inplace=True)
            if column == 0:
                column = 'global'
            df = df.iloc[1:]  # remove global row
            self.save(df, dim, f'aha_{dim}_{table_name}_{column}')

    def save(self, df: pd.DataFrame, dim: str, name: str) -> None:
        name = name.replace('/', '-')
        file_path = os.path.join(self.dst, dim, f'{name}.xlsx')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_excel(file_path, index=True)


if __name__ == '__main__':
    src = '/home/sebalzer/Documents/Mike_init/tests/train/6_condensed'
    dst = '/home/sebalzer/Documents/Mike_init/tests/train/7_merged'
    tm = MergeSegments(src, dst)

    # dims = ['2d', '3d']
    dims = ['3d']
    for dim in dims:
        for name in [
            'longit_strain_rate',
            'radial_strain_rate',
            'circumf_strain_rate',
            'longit_velocity',
            'radial_velocity',
            'circumf_velocity',
            'longit_acceleration',
            'radial_acceleration',
            'circumf_acceleration',
            'longit_strain-acc',
            'radial_strain-acc',
            'circumf_strain-acc',
        ]:
            tm(dim, name)
