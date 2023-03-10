import os
import sys

from loguru import logger
from omegaconf import OmegaConf

from feature_corr.factory_parts.data_reader import DataReader
from feature_corr.factory_parts.factory import Factory


def main():

    logger.info(f'Loading config file -> {os.path.join(os.getcwd(), "config.yaml")}')
    with open('config.yaml') as file:
        config = OmegaConf.load(file)

    with open('paths.yaml') as file:  # will be removed
        path = OmegaConf.load(file)

    config = OmegaConf.merge(config, path)

    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    DataReader(config)()
    factory = Factory(config)
    factory()


if __name__ == '__main__':
    main()
