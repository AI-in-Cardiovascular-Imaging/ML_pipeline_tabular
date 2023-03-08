from loguru import logger
from omegaconf import DictConfig

from cardio_parsers.crates.inspections import CleanUp, TargetStatistics
from cardio_parsers.factory_parts.pipeline import Pipeline
from cardio_parsers.factory_parts.state_machine import StateMachine


class Factory:
    """Produces pipelines"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.state_machine = StateMachine(config)

    def __call__(self) -> None:
        """Run factory"""
        logger.info('Factory started')

        TargetStatistics(self.config).show_target_statistics()
        CleanUp(self.config)()

        for config in self.state_machine:
            self.produce_pipeline(config)

    @staticmethod
    def produce_pipeline(config: DictConfig) -> None:
        """Pipeline producer"""
        pipeline = Pipeline(config)
        pipeline()
