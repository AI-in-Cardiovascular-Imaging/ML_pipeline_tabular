""" Pre-processing module with the ability to extract excel files ready for data analysis
    from raw civ42 data excel files
"""

import os

import hydra
from loguru import logger
from omegaconf import DictConfig

from excel.global_helpers import checked_dir
from excel.pre_processing.utils.workbook_2_sheets import ExtractWorkbook2Sheets
from excel.pre_processing.utils.sheets_2_tables import ExtractSheets2Tables
from excel.pre_processing.utils.cleaner import TableCleaner
from excel.pre_processing.utils.checks import SplitByCompleteness
from excel.pre_processing.utils.helpers import SaveTables


class Preprocessing:
    def __init__(self, config: DictConfig) -> None:
        self.src_dir = config.dataset.raw_dir
        self.dst_dir = config.dataset.out_dir
        self.save_intermediate = config.dataset.save_intermediate
        self.save_final = config.dataset.save_final
        self.dims = config.dataset.dims
        self.strict = config.dataset.strict

        self.dir_name = checked_dir(self.dims, self.strict)

    def __call__(self) -> None:
        if self.save_intermediate:
            logger.info('Intermediate results will be saved between each pre-processing step.')
            dst = os.path.join(self.dst_dir, '1_extracted')
        else:
            dst = os.path.join(self.dst_dir, '4_checked', self.dir_name)

        # Extract one sheet per patient from the available raw workbooks
        # additionally removes any colour formatting
        sheets = {}
        for src_file in os.listdir(src_dir):
            # for src_file in [os.path.join(src_dir, 'D. Strain_v3b_FlamBer_61-120.xlsx')]:
            if src_file.endswith('.xlsx') and not src_file.startswith('.'):
                logger.info(f'File -> {src_file}')
                workbook_2_sheets = ExtractWorkbook2Sheets(
                    src=os.path.join(src_dir, src_file), dst=dst, save_intermediate=self.save_intermediate
                )
                sheets = sheets | workbook_2_sheets()

                if self.save_intermediate:  # update paths
                    src_dir = dst
                    dst = os.path.join(self.dst_dir, '2_case_wise')

                sheets_2_tables = ExtractSheets2Tables(
                    src=src_dir, dst=dst, save_intermediate=self.save_intermediate, sheets=sheets
                )
                tables = sheets_2_tables()

                if self.save_intermediate:  # update paths
                    src_dir = dst
                    dst = os.path.join(self.dst_dir, '3_cleaned')

                cleaner = TableCleaner(
                    src=src_dir,
                    dst=dst,
                    save_intermediate=self.save_intermediate,
                    dims=self.dims,
                    tables=tables,
                    strict=self.strict,
                )
                clean_tables = cleaner()

                if self.save_intermediate:  # update paths
                    src_dir = dst
                    dst = os.path.join(self.dst_dir, '4_checked', self.dir_name)

                checker = SplitByCompleteness(
                    src=src_dir,
                    dst=dst,
                    save_intermediate=self.save_intermediate,
                    dims=self.dims,
                    tables=clean_tables,
                    strict=self.strict,
                )
                complete_tables = checker()

                # Save final pre-processed tables (only relevant if save_intermediate=False)
                if not self.save_intermediate and self.save_final:
                    saver = SaveTables(dst=dst, tables=complete_tables)

                    saver()


if __name__ == '__main__':

    @hydra.main(version_base=None, config_path='../../config', config_name='config')
    def main(config: DictConfig) -> None:
        pre_processing = Preprocessing(config)
        pre_processing()

    main()
