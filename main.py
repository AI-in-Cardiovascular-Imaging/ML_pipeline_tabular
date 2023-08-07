import os
import sys

from loguru import logger

from feature_corr.config_manager import ConfigManager
from feature_corr.crates.inspections import CleanUp, TargetStatistics
from feature_corr.factory_parts.data_reader import DataReader
from feature_corr.factory_parts.factory import Factory
from feature_corr.factory_parts.report import Report


def main(config_file: str = None) -> None:
    """Main function"""

    cwd = os.path.abspath(os.getcwd())
    config = ConfigManager(config_file, cwd)()

    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    if not config.meta.overwrite:
        DataReader(config)()
        CleanUp(config)()
        TargetStatistics(config).show_target_statistics()
        report = Report(config)
        Factory(config, report)()
    else:  # summarise already calculated results stored in json files
        Report(config)()

if __name__ == '__main__':
    main()
