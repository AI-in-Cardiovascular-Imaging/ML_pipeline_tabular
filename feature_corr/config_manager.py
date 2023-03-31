import os
import shutil
import sys

from loguru import logger
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    def __init__(self, config_file: str = None, cwd: str = '') -> None:
        self.config_file = config_file
        self.cwd = cwd

        logger.remove()
        logger.add(sys.stderr, level='INFO')

    def __call__(self) -> DictConfig:
        config = self._load_config_file()
        config = self._none_checks(config)
        return config

    def _load_config_file(self) -> DictConfig:
        """Load config file and merge with paths file"""
        if self.config_file is None:
            load_path = os.path.join(self.cwd, 'config.yaml')
        else:
            load_path = self.config_file

        logger.info(f'Try to load config file -> {load_path}')
        if not os.path.exists(load_path):
            shutil.copy(os.path.join(os.path.dirname(__file__), 'config.yaml'), os.path.join(load_path))
            logger.warning(f'Could not find config file, new config file was created -> {load_path}')
            exit(0)

        try:
            with open(load_path, 'r', encoding='utf-8') as file:
                config = OmegaConf.load(file)
        except Exception as e:
            raise KeyError(f'Type error in config file -> \n{e}')

        return config

    @staticmethod
    def _none_checks(config: DictConfig) -> DictConfig:
        """Check if that config file contains 'None' as string, this is not allowed, use null instead"""

        def check_for_string_none(dictionary: DictConfig) -> DictConfig or None:
            for key, value in dictionary.items():
                if isinstance(value, DictConfig):
                    check_for_string_none(value)
                elif isinstance(value, str) and value.lower() == 'none':
                    raise ValueError(f'None is not allowed, use null instead in config file -> {key}')

        check_for_string_none(config)
        return config
