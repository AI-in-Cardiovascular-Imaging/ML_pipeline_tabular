import os
import shutil
import sys

from loguru import logger
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    def __init__(self) -> None:
        self.config = None
        self.cwd = os.path.abspath(os.getcwd())
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    def __call__(self) -> DictConfig:
        self.load_config_file()
        self.range_to_list()
        OmegaConf.save(
                self.config, os.path.join(self.config.meta.output_dir, self.config.meta.name, 'job_config.yaml')
            )  # save copy of config for future reference
        return self.config

    def load_config_file(self) -> DictConfig:
        """Load config file and merge with paths file"""
        load_path = os.path.join(self.cwd, 'config.yaml')

        logger.info(f'Try to load config file -> {load_path}')
        if not os.path.exists(load_path):
            shutil.copy(os.path.join(os.path.dirname(__file__), 'config.yaml'), os.path.join(load_path))
            logger.warning(f'Could not find config file, new config file was created -> {load_path}')
            exit(0)

        try:
            with open(load_path, 'r', encoding='utf-8') as file:
                self.config = OmegaConf.load(file)
        except Exception as e:
            raise KeyError(f'Type error in config file -> \n{e}')

    def range_to_list(self):
        """Convert range string to list"""
        if isinstance(self.config.verification.use_n_top_features, str):
            self.config.verification.use_n_top_features = list(eval(self.config.verification.use_n_top_features))