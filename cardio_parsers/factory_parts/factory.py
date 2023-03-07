from factory_parts.data_reader import DataReader
from factory_parts.pipeline import Pipeline
from factory_parts.state_machine import StateMachine
from loguru import logger
from omegaconf import DictConfig

from cardio_parsers.data_borg.data_borg import DataBorg


class Factory:
    """Produces pipelines"""

    def __init__(self, config) -> None:
        self.config = config
        self.state_machine = StateMachine(config)
        self.data_reader = DataBorg()

    def __call__(self) -> None:
        """Run factory"""
        logger.info('Factory started')
        DataReader(self.config)()
        for config in self.state_machine:  # TODO: multiprocessing
            print(type(config))
            self.data_reader.add_state_name(config.meta.state_name)
            self.produce_pipeline(config)

    @staticmethod
    def produce_pipeline(config: DictConfig) -> None:
        """Pipeline producer"""
        pipeline = Pipeline(config)
        pipeline()
