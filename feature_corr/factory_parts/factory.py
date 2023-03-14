import os
import time

from loguru import logger
from omegaconf import DictConfig

from feature_corr.crates.inspections import CleanUp, TargetStatistics
from feature_corr.factory_parts.pipeline import Pipeline
from feature_corr.factory_parts.report import Report
from feature_corr.factory_parts.state_machine import StateMachine


class Factory:
    """Produces pipelines"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.report = Report()
        self.state_machine = StateMachine(config)

    def __call__(self) -> None:
        """Run factory"""
        logger.info('Factory started')

        CleanUp(self.config)()
        TargetStatistics(self.config).show_target_statistics()

        for config in self.state_machine:
            self.produce_pipeline(config)

    def __del__(self):
        self.report()
        logger.info('Factory finished')
        logger.info(f'Check results in -> {os.path.join(self.config.meta.output_dir, self.config.meta.name)}')

    @staticmethod
    def produce_pipeline(config: DictConfig) -> None:
        """Pipeline producer"""
        start_time = time.time()
        pipeline = Pipeline(config)
        pipeline()
        logger.info(f'Pipelines finished in {(time.time() - start_time)/60:.2f} minutes')
