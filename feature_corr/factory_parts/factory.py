import os
import time
import typing as t

from loguru import logger
from omegaconf import DictConfig

from feature_corr.factory_parts.pipeline import Pipeline
from feature_corr.factory_parts.state_machine import StateMachine


class Factory:
    """Produces pipelines"""

    def __init__(self, config: DictConfig, report: t.Callable) -> None:
        self.config = config
        self.report = report
        self.state_machine = StateMachine(config)

    def __call__(self) -> None:
        """Run factory"""
        logger.info('Factory started')
        for config in self.state_machine:
            self.produce_pipeline(config)
        self.report()
        logger.warning('Factory finished')

    def __del__(self):
        logger.info('Factory finished')
        logger.info(f'Check results in -> {os.path.join(self.config.meta.output_dir, self.config.meta.name)}')

    @staticmethod
    def produce_pipeline(config: DictConfig) -> None:
        """Pipeline producer"""
        start_time = time.time()
        pipeline = Pipeline(config)
        pipeline()
        logger.info(f'Pipelines finished in {(time.time() - start_time)/60:.2f} minutes')
