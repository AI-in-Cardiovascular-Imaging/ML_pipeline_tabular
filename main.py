import os
import sys
import warnings

from loguru import logger

from pipeline_tabular.config_manager import ConfigManager
from pipeline_tabular.utils.inspections import CleanUp, DataExploration
from pipeline_tabular.run.data_reader import DataReader
from pipeline_tabular.run.run import Run


def main() -> None:
    config = ConfigManager()()
    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)
    if config.meta.ignore_warnings:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    DataReader(config)()
    CleanUp(config)()
    DataExploration(config)()
    Run(config)()

if __name__ == '__main__':
    main()
