import sys

from factory_parts.factory import Factory
from loguru import logger
from omegaconf import OmegaConf

if __name__ == '__main__':

    with open('config.yaml') as file:
        config = OmegaConf.load(file)

    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    fa = Factory(config)
    fa()
