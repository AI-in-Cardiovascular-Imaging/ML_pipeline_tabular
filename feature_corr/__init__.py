import os
import sys

from loguru import logger

from feature_corr.config_manager import ConfigManager
from feature_corr.crates.inspections import CleanUp, TargetStatistics
from feature_corr.crates.verifications import Verification
from feature_corr.factory_parts.data_reader import DataReader
from feature_corr.factory_parts.factory import Factory
from feature_corr.factory_parts.report import Report


def start(config_file: str = None) -> None:
    """Main function"""

    cwd = os.path.abspath(os.getcwd())
    config = ConfigManager(config_file, cwd)()

    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    report = Report(config)
    DataReader(config)()
    CleanUp(config)()
    TargetStatistics(config).show_target_statistics()

    Factory(config, report)()

    # if config.meta.run_verification:
    #     top_features = report.get_rank_frequency_based_features()
    #     Verification(config, top_features)()
