"""Helper class to save tables stored in NestedDefaultDict to .xlsx files
"""

import os

from collections import defaultdict
from loguru import logger
import pandas as pd


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded on the fly"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(dict(self))


class SaveTables:
    """Save tables from NestedDefaultDict to .xlsx files"""

    def __init__(self, dst: str, dims: list = ['2d'], tables: NestedDefaultDict = None) -> None:
        self.dst = dst
        self.dims = dims
        self.tables = tables

    def __call__(self) -> None:
        logger.info('Saving tables...')
        for subject in list(self.tables.keys()):
            logger.info(f'Saving tables for subject {subject}')
            for dim in self.dims:
                for table_name, table in self.tables[subject][dim].items():
                    if table is None:
                        continue
                    self.save(table, subject, dim, table_name)

    def save(self, df: pd.DataFrame, subject: str, dim: str, table: str) -> None:
        """Save table"""
        export_path = os.path.join(self.dst, subject, dim, table)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        df.to_excel(f'{export_path}.xlsx', index=False)
