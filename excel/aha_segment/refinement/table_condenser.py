import os

import pandas as pd
from loguru import logger


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class TableCondenser:
    """Narrows down the table to the columns of interest"""

    def __init__(self, src: str, dst: str) -> None:
        self.src = src
        self.dst = dst
        self.memory = {}

    def __call__(self) -> None:
        dims = ['2d', '3d']
        # dims = ['2d']
        for dim in dims:
            for subject in self.loop_subjects():
                for table in self.loop_tables(subject, dim):
                    df = self.clean(subject, dim, table)
                    self.save(df, subject, dim, table)

    def loop_subjects(self) -> str:
        """Loop over subjects"""
        for subject in os.listdir(self.src):
            logger.info(f'-> {subject}')
            yield subject

    def loop_tables(self, subject: str, dim: str) -> str:
        """Loop over tables"""
        if os.path.exists(os.path.join(self.src, subject, dim)):
            for table in os.listdir(os.path.join(self.src, subject, dim)):
                if table.endswith('.xlsx'):
                    logger.info(f'-> {table}')
                    yield table

    def clean(self, subject: str, dim: str, table: str) -> pd.DataFrame or None:
        """Clean table"""
        table_path = os.path.join(self.src, subject, dim, table)
        df = pd.read_excel(table_path)

        if not df.empty:
            # keep only columns of interest
            # df = df[[col for col in df.columns if 'sample' in col or 'AHA' in col or 'slice' in col or 'roi' in col]]
            df = df[[col for col in df.columns if 'sample' in col or 'AHA' in col]]
            return df
        return None

    def save(self, df: pd.DataFrame, subject: str, dim: str, table: str) -> None:
        """Save table"""
        if df is not None:
            export_path = os.path.join(self.dst, subject, dim, table)
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            df.to_excel(export_path, index=False)


if __name__ == '__main__':
    src = os.path.join('/home/sebalzer/Documents/Mike_init/tests/train/4_checked', 'complete')
    dst = '/home/sebalzer/Documents/Mike_init/tests/train/6_condensed'
    tc = TableCondenser(src, dst)
    tc()
