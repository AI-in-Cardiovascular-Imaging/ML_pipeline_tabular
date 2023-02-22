""" Pre-processing module with the ability to extract excel files ready for data analysis
    from raw civ42 data excel files
"""

import os
import sys

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

        # Extract one sheet per patient from the available raw workbooks
        # additionally removes any colour formatting
        sheets = {}
        for dir in os.listdir(self.src_dir):
            if dir == 'redcap_id':
                suffix = '_rc'
            elif dir == 'pat_id':
                suffix = '_p'
            else:
                logger.error('Unknown data source, must be either redcap_id or pat_id.')
                raise NotImplementedError

            for src_file in os.listdir(os.path.join(self.src_dir, dir)):
                if src_file.endswith('.xlsx') and not src_file.startswith('.'):
                    if self.save_intermediate:
                        logger.info('Intermediate results will be saved between each pre-processing step.')
                        dst = os.path.join(self.dst_dir, '1_extracted')
                    else:
                        dst = os.path.join(self.dst_dir, '4_checked', self.dir_name)
                        
                    logger.info(f'File -> {os.path.join(dir, src_file)}')
                    workbook_2_sheets = ExtractWorkbook2Sheets(
                        src=os.path.join(self.src_dir, dir, src_file),
                        dst=dst,
                        suffix=suffix,
                        save_intermediate=self.save_intermediate,
                    )
                    sheets = sheets | workbook_2_sheets()

                    if self.save_intermediate:  # update paths
                        src = dst
                        dst = os.path.join(self.dst_dir, '2_case_wise')
                    else:
                        src = self.src_dir

                    sheets_2_tables = ExtractSheets2Tables(
                        src=src, dst=dst, save_intermediate=self.save_intermediate, sheets=sheets
                    )
                    tables = sheets_2_tables()

                    if self.save_intermediate:  # update paths
                        src = dst
                        dst = os.path.join(self.dst_dir, '3_cleaned')

                    cleaner = TableCleaner(
                        src=src,
                        dst=dst,
                        save_intermediate=self.save_intermediate,
                        dims=self.dims,
                        tables=tables,
                        strict=self.strict,
                    )
                    clean_tables = cleaner()

                    if self.save_intermediate:  # update paths
                        src = dst
                        dst = os.path.join(self.dst_dir, '4_checked', self.dir_name)

                    checker = SplitByCompleteness(
                        src=src,
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
        logger.remove()
        logger.add(sys.stderr, level=config.logging_level)
        pre_processing = Preprocessing(config)
        pre_processing()

    main()
