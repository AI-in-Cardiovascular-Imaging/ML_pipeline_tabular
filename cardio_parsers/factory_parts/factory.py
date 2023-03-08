from factory_parts.pipeline import Pipeline
from factory_parts.state_machine import StateMachine
from loguru import logger
from omegaconf import DictConfig


class Factory:
    """Produces pipelines"""

    def __init__(self, config) -> None:
        self.config = config
        self.state_machine = StateMachine(config)

    def __call__(self) -> None:
        """Run factory"""
        logger.info('Factory started')

        # TargetStatistics(self.config)()
        # CleanUp(self.config)()

        for config in self.state_machine:  # TODO: multiprocessing
            self.produce_pipeline(config)

    @staticmethod
    def produce_pipeline(config: DictConfig) -> None:
        """Pipeline producer"""
        pipeline = Pipeline(config)
        pipeline()
