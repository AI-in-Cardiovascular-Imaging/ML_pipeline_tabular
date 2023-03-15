import os
import sys

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from feature_corr.crates.inspections import CleanUp, TargetStatistics
from feature_corr.crates.verifications import Verification
from feature_corr.factory_parts.data_reader import DataReader
from feature_corr.factory_parts.factory import Factory
from feature_corr.factory_parts.report import Report


def main() -> None:
    """Main function"""
    config = load_config_file()

    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)

    verification_only = config.meta.verification_only
    selection_only = config.meta.selection_only

    report = Report(config)
    DataReader(config)()
    CleanUp(config)()

    if not verification_only:
        TargetStatistics(config).show_target_statistics()
        factory = Factory(config, report)
        factory()

    if not selection_only:
        top_features = report.get_rank_frequency_based_features()
        Verification(config, top_features).verify_final()


def load_config_file() -> DictConfig:
    """Load config file and merge with paths file"""
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

    return OmegaConf.merge(config, path)


if __name__ == '__main__':
    main()
