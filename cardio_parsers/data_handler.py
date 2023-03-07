from collections import defaultdict

from loguru import logger
from omegaconf import DictConfig, OmegaConf


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded"""

    def __init__(self, *args, **kwargs):
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


class DataHandler:

    _shared_state = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_state
        self.store = NestedDefaultDict()

    def __call__(self):
        logger.info(self.store)

    def __del__(self) -> None:
        logger.info('DataHandler stopped')

    # def __str__(self) -> DictConfig:
    #
    #     logger.info(f'{self.store})')

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __contains__(self, key):
        return key in self.store
