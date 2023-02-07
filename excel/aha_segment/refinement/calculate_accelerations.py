import os

import pandas as pd
from loguru import logger

from excel.global import CHECKED_PATH, CONDENSED_PATH

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# TODO: Need to check Jerk as well
# TODO: only done for 3d so far (hardcoded)
class CalculateAcceleration:
    """Narrows down the table to the columns of interest"""

    def __init__(self, src: str, dst: str) -> None:
        self.src = src
        self.dst = dst
        self.memory = {}

    def __call__(self) -> None:
        for subject in self.loop_subjects():
            for name in ['strain_rate']:
                for table in self.loop_tables(subject, name):
                    df = self.get_acceleration(subject, table)
                    self.save(df, subject, name, table)
            for name in ['velocity']:
                for table in self.loop_tables(subject, name):
                    df = self.get_acceleration(subject, table)
                    self.save(df, subject, name, table)

    def loop_subjects(self) -> str:
        """Loop over subjects"""
        for subject in os.listdir(self.src):
            logger.info(f'-> {subject}')
            yield subject

    def loop_tables(self, subject: str, name: str) -> str:
        """Loop over tables"""
        if os.path.exists(os.path.join(self.src, subject, '3d')):
            for table in os.listdir(os.path.join(self.src, subject, '3d')):
                if name in table:
                    logger.info(f'-> {table}')
                    yield table

    def get_acceleration(self, subject: str, table: str) -> pd.DataFrame or None:
        """Get acceleration"""
        table_path = os.path.join(self.src, subject, '3d', table)
        df = pd.read_excel(table_path)
        df_acc = pd.DataFrame()
        df_acc['AHA Segment'] = df['AHA Segment']
        for col in df.columns:
            if 'sample' in col:
                i = int(col.split('_')[-1])  # get sample number
                if i >= 23:  # last sample has no acceleration
                    continue
                delta_v = df[f'sample_{i + 2}'] - df[f'sample_{i + 1}']  # mm/s
                delta_t = df[f'time_{i + 2}'] - df[f'time_{i + 1}']  # ms
                delta_t = delta_t / 1000  # convert ms to s
                df_acc[f'sample_{i}'] = delta_v / delta_t
        return df_acc

    def save(self, df: pd.DataFrame, subject: str, name: str, table: str) -> None:
        """Save table"""
        if df is not None:
            if name == 'velocity':
                tag = 'acceleration'
            else:
                tag = 'strain-acc'
            table = table.replace(name, tag)
            table = table.replace('-s', '-s^2')
            export_path = os.path.join(self.dst, subject, '3d', table)
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            df.to_excel(export_path, index=False)


if __name__ == '__main__':
    src = os.path.join(CHECKED_PATH, 'complete')
    dst = CONDENSED_PATH
    ca = CalculateAcceleration(src, dst)
    ca()
