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
        logger.trace(f'Features -> {json.dumps(self.all_features, indent=4)}')

    def get_rank_frequency_based_features(self) -> list:
        """Get ranked features"""
        if self.all_features is None:
            self.load_features()

        store = {}
        for state_name in self.all_features.keys():
            for job_name in self.all_features[state_name].keys():
                rank_score = 1000
                for features in self.all_features[state_name][job_name]:
                    for feature in [features]:
                        if feature not in store:
                            store[feature] = rank_score
                        else:
                            store[feature] += rank_score
                        rank_score -= 1

        sorted_store = {k: v for k, v in sorted(store.items(), key=lambda item: item[1], reverse=True)}
        sorted_store = list(sorted_store.keys())
        return_top = self.config.verification.use_n_top_features
        top_features = sorted_store[:return_top]
        logger.info(
            f'Rank frequency based top {min(return_top, len(top_features))} features -> {json.dumps(top_features, indent=4)}'
        )
        return top_features
