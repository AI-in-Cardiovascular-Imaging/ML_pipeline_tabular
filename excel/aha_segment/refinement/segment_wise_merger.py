import os

import pandas as pd
from loguru import logger

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class MergeCasesOfPolarMaps:
    """Merge table of subjects"""

    def __init__(self, src: str, dst: str) -> None:
        self.src = src
        self.dst = dst
        self.memory = {}

    def __call__(self) -> None:
        subjects = os.listdir(self.src)
        subject_path = os.path.join(self.src, subjects[0])
        tables = os.listdir(subject_path)
        tables = ['_'.join(table.split('_')[1:]) for table in tables]

        for table in tables:
            if 'polarmap' in table:
                logger.info(f'-> {table}')
                table_name = table.replace('.xlsx', '')
                self.memory = {}
                for subject in subjects:
                    file_name = f'{subject}_{table}'
                    file_path = os.path.join(self.src, subject, file_name)
                    df = pd.read_excel(file_path)
                    self.memory[subject] = df
                self.merge_column_wise(table_name)

    def merge_column_wise(self, table_name) -> None:
        columns = self.memory[list(self.memory.keys())[0]].columns
        for column in columns:
            df = pd.DataFrame(columns=self.memory.keys())
            for subject in self.memory:
                df[subject] = self.memory[subject][column]

            table_name = table_name.replace('/', '-')
            column = column.replace('/', '-')
            file_path = os.path.join(self.dst, table_name, f'{table_name}_{column}.xlsx')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_excel(file_path, index=False)


if __name__ == '__main__':
    src = '/home/melandur/Downloads/tables_without_index/'
    dst = '/home/melandur/Downloads/new'
    tm = MergeCasesOfPolarMaps(src, dst)
    tm()
