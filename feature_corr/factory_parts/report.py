import json
import os

from loguru import logger
from omegaconf import DictConfig

from feature_corr.data_borg import DataBorg


class Report(DataBorg):
    """What's my purpose? You are passing the features"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.all_features = None
        experiment_name = self.config.meta.name
        output_dir = self.config.meta.output_dir
        self.feature_file_path = os.path.join(output_dir, experiment_name, 'all_features.json')

    def __call__(self):
        """Run feature report"""
        all_features = self.get_all_features()
        if all_features:
            self.write_to_file(all_features)
        else:
            logger.warning('No features found to report')

    def write_to_file(self, all_features: dict) -> None:
        """Write features to file"""
        with open(self.feature_file_path, 'w+', encoding='utf-8') as file:
            json.dump(all_features, file, indent=4)

        with open(self.feature_file_path, 'r', encoding='utf-8') as file:
            loaded_features = json.load(file)

        if loaded_features != all_features:
            logger.warning(f'Failed to write features -> {self.feature_file_path}')
        else:
            logger.info(f'Saved features to -> {self.feature_file_path}')

    def load_features(self) -> None:
        """Load features from file"""
        if not os.path.exists(self.feature_file_path):
            raise FileNotFoundError(f'Could not find features file -> {self.feature_file_path}')
        logger.info(f'Loading features from -> {self.feature_file_path}')
        with open(self.feature_file_path, 'r', encoding='utf-8') as file:
            self.all_features = json.load(file)

    def get_rank_frequency_based_features(self, number: int = 10) -> list:
        """Get ranked features"""
        if self.all_features is None:
            self.load_features()
        return self.all_features


if __name__ == '__main__':
    from omegaconf import OmegaConf

    conf = OmegaConf.create({'meta': {'name': 'test', 'output_dir': '/home/melandur/Downloads'}})
    r = Report(conf)
    r.get_rank_frequency_based_features()
