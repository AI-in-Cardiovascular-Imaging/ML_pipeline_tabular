from loguru import logger


class DataHandler:

    _shared_state = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_state
        self.store = {}

    def __call__(self):
        logger.info(self.store)
