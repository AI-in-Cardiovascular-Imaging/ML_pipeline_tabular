from loguru import logger
from omegaconf import OmegaConf

from feature_corr.crates.imputers import Imputers
from feature_corr.crates.inspections import TargetStatistics
from feature_corr.data_borg import DataBorg


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
        self.sync_ephemeral_data_to_data_store(self.state_name, 'ephemeral')

    def __call__(self) -> None:
        """Iterate over pipeline steps"""
        for step in self.config.keys():
            getattr(self, step)()

    def __del__(self):
        """Delete assigned state data"""
        self.remove_state_data_store(self.state_name)

    @staticmethod
    def meta() -> None:
        """Skip meta step"""

    def inspection(self) -> None:
        """Skip inspection step"""
        learning_task = TargetStatistics(self.config).set_target_task()
        OmegaConf.update(self.config.selection, 'learn_task', learning_task)

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
