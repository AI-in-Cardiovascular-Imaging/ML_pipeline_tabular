from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cardio_parsers.data_handler import DataHandler
from cardio_parsers.data_reader import DataReader
from cardio_parsers.pipeline import Pipeline
from cardio_parsers.state_machine import StateMachine


class Factory:
    """Produces pipelines"""

    def __init__(self) -> None:
        self.config = OmegaConf.load('config.yaml')
        self.state_machine = StateMachine(self.config)
        self.data_handler = DataHandler()

    def __del__(self) -> None:
        self.data_handler()
        logger.info('Factory stopped')

    def __call__(self) -> None:
        """Run factory"""
        logger.info('Factory started')
        DataReader(self.config.meta.input_file)()
        for state, config in self.state_machine:
            self.produce_pipeline(state, config)

    @staticmethod
    def produce_pipeline(state: str, config: DictConfig) -> None:
        """Pipeline producer"""
        pipeline = Pipeline(state, config)
        pipeline()
