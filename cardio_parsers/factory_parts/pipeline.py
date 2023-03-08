from loguru import logger

from cardio_parsers.crates.imputers import Imputers
from cardio_parsers.data_borg.data_borg import DataBorg


def run_when_active(func):
    """Decorator to run pipeline step when active"""

    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        if self.config[func_name]['active']:
            logger.info(f'Running -> {func_name}')
            func(self, *args, **kwargs)

    return wrapper


class Pipeline(DataBorg):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.state_name = config.meta.state_name
        self.add_state_name(self.state_name)
        self.copy_original_to_ephemeral(self.state_name)

    def __call__(self) -> None:
        """Iterate over pipeline steps"""
        for step in self.config.keys():
            getattr(self, step)()

    def __del__(self):
        """Delete assigned state data"""
        self.remove_state_data(self.state_name)

    @staticmethod
    def meta() -> None:
        """Skip meta step"""

    @staticmethod
    def inspection() -> None:
        """Skip inspection step"""

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
