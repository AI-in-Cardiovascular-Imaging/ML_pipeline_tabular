import sys

from loguru import logger
from omegaconf import OmegaConf

from cardio_parsers.factory_parts.data_reader import DataReader
from cardio_parsers.factory_parts.factory import Factory

with open('config.yaml') as file:
    config = OmegaConf.load(file)

logger.remove()
logger.add(sys.stderr, level=config.meta.logging_level)

DataReader(config)()
fa = Factory(config)
fa()
