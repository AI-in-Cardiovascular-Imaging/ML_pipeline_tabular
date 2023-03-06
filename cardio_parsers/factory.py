from loguru import logger
from omegaconf import OmegaConf

from cardio_parsers.data_handler import DataHandler
from cardio_parsers.pipeline import Pipeline
from cardio_parsers.state_machine import StateMachine


class Factory:
    def __init__(self) -> None:
        self.config = OmegaConf.load('config.yaml')
        self.sm = StateMachine(self.config)
        self.dh = DataHandler()

    def __del__(self):
        self.dh()
        logger.info('Factory stopped')

    def __call__(self) -> None:
        """Run factory"""
        for state, config in self.sm:
            logger.info(f'->: {state}')
            self.produce_pipeline(state, config)

    @staticmethod
    def produce_pipeline(state, config):
        """Run pipeline"""
        logger.info(f'->: {state}')
        pipeline = Pipeline(state, config)
        pipeline()
