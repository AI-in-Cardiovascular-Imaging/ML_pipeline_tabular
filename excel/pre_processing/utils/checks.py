import os
import shutil

from loguru import logger
import pandas as pd

from excel.pre_processing.utils.helpers import NestedDefaultDict


class SplitByCompleteness:
    """Sort files by completeness"""

    def __init__(
        self,
        src: str,
        dst: str,
        save_intermediate: bool = True,
        dims: list = ['2d'],
        tables: NestedDefaultDict = None,
        strict: bool = False,
    ) -> None:
        self.src = src
        self.dst = dst
        self.save_intermediate = save_intermediate
        self.dims = dims
        self.tables = tables
        self.strict = strict

        self.count = 0
        self.memory = {}
        self.complete_files = {}
        self.missing_files = {}

        # Set target count according to requested dims
        if not self.strict:
            self.target_count = 1
        elif '2d' in dims and '3d' in dims:
            self.target_count = 45
        elif '2d' in dims:
            self.target_count = 32
        elif '3d' in dims:
            self.target_count = 13
        else:
            logger.error('dims must contain 2d or 3d, check your config.yaml file')
            raise NotImplementedError

    def __call__(self) -> NestedDefaultDict:
        if self.save_intermediate:
            for case in self.get_cases():
                self.count_files(case)
                self.memory[case] = self.count
            self.divide_cases()
            self.move_files()

        else:  # use dict of DataFrames
            # Instead of dividing into complete and missing,
            # delete all subjects with missing tables in requested dims
            for subject in list(self.tables.keys()):
                logger.info(f'Checking subject -> {subject}')
                # Check whether sufficient tables are present
                count = 0
                for dim in self.dims:
                    count += len(self.tables[subject][dim])
                if count < self.target_count:
                    del self.tables[subject]
                    logger.info(f'Removed subject {subject} due to missing tables.')
                # else:
                #     logger.info(f'Complete subject -> {subject}')

        return self.tables

    def get_cases(self) -> str:
        """Get all cases from the cleaned folder"""
        cases = os.listdir(self.src)
        for case in cases:
            logger.info(f'Checking subject -> {case}')
            self.count = 0
            yield case

    def count_files(self, case: str) -> None:
        """Count the number of files in a case folder"""
        for root, _, files in os.walk(os.path.join(self.src, case)):
            for file in files:
                if file.endswith('.xlsx'):
                    df = pd.read_excel(os.path.join(root, file))
                    if not df.iloc[:, 5].isnull().all():  # checks column 5 for NaN
                        self.count += 1

    def divide_cases(self) -> None:
        """Divide cases into complete and missing"""
        for case, counted_files in self.memory.items():
            if counted_files < self.target_count:
                self.missing_files[case] = counted_files
            else:
                self.complete_files[case] = counted_files

    def move_files(self) -> None:
        """Move files to their destination folder"""
        logger.info('Copy complete cases')
        for case in self.complete_files:
            complete_file_path = os.path.join(self.dst, case)
            os.makedirs(complete_file_path, exist_ok=True)
            shutil.copytree(os.path.join(self.src, case), complete_file_path, dirs_exist_ok=True)
            logger.info(f'Complete subject -> {case}')
