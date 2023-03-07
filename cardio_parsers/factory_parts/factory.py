from data_borg.data_borg import DataBorg
from factory_parts.data_reader import DataReader
from factory_parts.pipeline import Pipeline
from factory_parts.state_machine import StateMachine
from loguru import logger
from omegaconf import DictConfig, OmegaConf


class Factory:
    """Produces pipelines"""

    def __init__(self) -> None:
        self.config = OmegaConf.load('config.yaml')
        self.state_machine = StateMachine(self.config)
        self.data_handler = DataBorg()

    def __del__(self) -> None:
        """Stop factory"""
        logger.info('Factory stopped')

    def __call__(self) -> None:
        """Run factory"""
        logger.info('Factory started')
        DataReader(self.config.meta.input_file)()
        for state, config in self.state_machine:  # TODO: multiprocessing
            self.produce_pipeline(state, config)

    @staticmethod
    def produce_pipeline(state: str, config: DictConfig) -> None:
        """Pipeline producer"""
        pipeline = Pipeline(state, config)
        pipeline()
