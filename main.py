import os
import sys

from loguru import logger

from feature_corr.config_manager import ConfigManager
from feature_corr.utils.inspections import CleanUp, TargetStatistics
from feature_corr.pipeline.data_reader import DataReader
from feature_corr.pipeline.pipeline import Pipeline


def main(config_file: str = None) -> None:
    """Main function"""

    cwd = os.path.abspath(os.getcwd())
    config = ConfigManager(config_file, cwd)()

    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    DataReader(config)()
    CleanUp(config)()
    TargetStatistics(config).show_target_statistics()
    Pipeline(config)()

if __name__ == '__main__':
    main()
