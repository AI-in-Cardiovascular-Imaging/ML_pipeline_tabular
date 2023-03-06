from loguru import logger

from cardio_parsers.data_handler import DataHandler


class Pipeline(DataHandler):
    def __init__(self, state, config) -> None:
        super().__init__()
        self.__dict__ = self._shared_state

        self.state = state
        self.config = config

    def __call__(self) -> None:
        for step in self.config.keys():
            getattr(self, step)()

    def experiment(self):
        logger.info('Running experiment...')

    def impute(self):
        logger.info('Imputing data...')

    def data_split(self):
        logger.info('Splitting data...')

    def exploration(self):
        logger.info('Exploring data...')

    def verification(self):
        logger.info('Verifying data...')
