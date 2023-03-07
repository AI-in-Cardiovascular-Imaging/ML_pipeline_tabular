import sys

from factory_parts.factory import Factory
from loguru import logger
from omegaconf import OmegaConf

if __name__ == '__main__':

    config = OmegaConf.load('config.yaml')
    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    fa = Factory(config)
    fa()
