from loguru import logger

from cardio_parsers.crates.imputers import Imputers
from cardio_parsers.crates.inspections import CleanUp, TargetStatistics


def run_when_active(func):
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        if self.config[func_name]['active']:
            logger.info(f'Running -> {func_name}')
            func(self, *args, **kwargs)

    return wrapper


class Pipeline:
    def __init__(self, state, config) -> None:
        self.state = state
        self.config = config

    def __call__(self) -> None:
        """Iterate over pipeline steps"""
        for step in self.config.keys():
            getattr(self, step)()

    @staticmethod
    def meta() -> None:
        """Maybe add some experiment info"""

    @run_when_active
    def inspection(self) -> None:
        """Inspect data"""
        TargetStatistics(self.config)()
        CleanUp(self.config)()

    @run_when_active
    def impute(self) -> None:
        """Impute data"""
        Imputers(self.config)()

    @run_when_active
    def data_split(self) -> None:
        """Split data"""

    @run_when_active
    def selection(self) -> None:
        """Explore data"""

    @run_when_active
    def verification(self) -> None:
        """Verify data"""
