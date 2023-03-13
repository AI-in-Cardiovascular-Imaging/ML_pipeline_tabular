import os
import sys

from loguru import logger
from omegaconf import OmegaConf

from feature_corr.factory_parts.data_reader import DataReader
from feature_corr.factory_parts.factory import Factory


def main():

    logger.info(f'Loading config file -> {os.path.join(os.getcwd(), "config.yaml")}')

    if not os.path.exists('config.yaml'):
        logger.error('Could not find config.yaml')
        sys.exit(1)

    if not os.path.exists('paths.yaml'):
        logger.error('Could not find paths.yaml')
        sys.exit(1)

    try:
        with open('config.yaml') as file:
            config = OmegaConf.load(file)

        with open('paths.yaml') as file:  # will be removed
            path = OmegaConf.load(file)
    except Exception as e:
        logger.error(f'Type error in config file -> \n{e}')
        sys.exit(1)

    config = OmegaConf.merge(config, path)

    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    DataReader(config)()
    factory = Factory(config)
    factory()


if __name__ == '__main__':
    main()
