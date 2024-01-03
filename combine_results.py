import os
import sys
import warnings

import pandas as pd
from loguru import logger

from pipeline_tabular.config_manager import ConfigManager


class CombineResults:
    def __init__(self, config):
        self.config = config
        self.out_dir = config.meta.output_dir
        self.to_collect = config.collect_results.to_collect

    def __call__(self) -> None:
        self.collect_data()

    def collect_data(self):
        for name in self.to_collect:
            best_scores = pd.read_csv(os.path.join(self.out_dir, name, 'report', 'best_model_all_scores.csv'))


if __name__ == '__main__':
    config = ConfigManager()()
    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)
    if config.meta.ignore_warnings:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    CombineResults(config)()
