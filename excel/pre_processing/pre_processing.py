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


@hydra.main(version_base=None, config_path='../../config', config_name='config')
def pre_processing(config: DictConfig) -> None:
    """Pre-processing pipeline

    Args:
        config (DictConfig): config element containing all config parameters
            check the config files for info on the individual parameters

    Returns:
        None
    """
    # Parse some config parameters
    src_dir = config.dataset.raw_dir
    dst_dir = config.dataset.out_dir
    save_intermediate = config.dataset.save_intermediate
    save_final = config.dataset.save_final
    dims = config.dataset.dims
    strict = config.dataset.strict

    dir_name = checked_dir(dims, strict)

    if save_intermediate:
        logger.info('Intermediate results will be saved between each pre-processing step.')
        dst = os.path.join(dst_dir, '1_extracted')
    else:
        dst = os.path.join(dst_dir, '4_checked', dir_name)

    # Extract one sheet per patient from the available raw workbooks
    # additionally removes any colour formatting
    sheets = {}
    for src_file in os.listdir(src_dir):
        # for src_file in [os.path.join(src_dir, 'D. Strain_v3b_FlamBer_61-120.xlsx')]:
        if src_file.endswith('.xlsx') and not src_file.startswith('.'):
            logger.info(f'File -> {src_file}')
            workbook_2_sheets = ExtractWorkbook2Sheets(
                src=os.path.join(src_dir, src_file), dst=dst, save_intermediate=save_intermediate
            )
            sheets = sheets | workbook_2_sheets()

            if save_intermediate:  # update paths
                src_dir = dst
                dst = os.path.join(dst_dir, '2_case_wise')

            sheets_2_tables = ExtractSheets2Tables(
                src=src_dir, dst=dst, save_intermediate=save_intermediate, sheets=sheets
            )
            tables = sheets_2_tables()

            if save_intermediate:  # update paths
                src_dir = dst
                dst = os.path.join(dst_dir, '3_cleaned')

            cleaner = TableCleaner(
                src=src_dir, dst=dst, save_intermediate=save_intermediate, dims=dims, tables=tables, strict=strict
            )
            clean_tables = cleaner()

            if save_intermediate:  # update paths
                src_dir = dst
                dst = os.path.join(dst_dir, '4_checked', dir_name)

            checker = SplitByCompleteness(
                src=src_dir, dst=dst, save_intermediate=save_intermediate, dims=dims, tables=clean_tables, strict=strict
            )
            complete_tables = checker()

            # Save final pre-processed tables (only relevant if save_intermediate=False)
            if not save_intermediate and save_final:
                saver = SaveTables(dst=dst, tables=complete_tables)

                saver()


if __name__ == '__main__':
    pre_processing()
