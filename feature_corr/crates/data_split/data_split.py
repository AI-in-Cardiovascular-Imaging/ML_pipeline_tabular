import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from feature_corr.data_handler import DataHandler


class DataSplit(DataHandler):
    """Split frame in selection and verification"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.seed = config.meta.seed
        self.state_name = config.meta.state_name
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.n_bootstraps = config.data_split.n_bootstraps
        self.test_frac = config.data_split.test_frac

        self.stratify = None
        self.frame = self.get_store('frame', self.state_name, 'ephemeral')

    def __call__(self, boot_seed):
        """Split data"""
        self.boot_seed = boot_seed
        self.split_frame()

    def split_frame(self) -> None:
        self.set_stratification(self.frame)
        train, test = self.create_split()

        if self.test_frac <= 0.0 or self.test_frac >= 1.0:
            raise ValueError('"test_frac" is invalid, must be float between (0.0, 1.0)')

        self.set_store('frame', self.state_name, 'train', train)
        self.set_store('frame', self.state_name, 'test', test)
        all_features = list(train.columns.drop(self.target_label))
        self.set_store('feature', self.state_name, 'all_features', all_features)

    def set_stratification(self, frame: pd.DataFrame = None) -> None:
        """Set stratification"""
        if self.learn_task == 'binary_classification':
            target_frame = frame[self.target_label]
            self.stratify = target_frame
        elif self.learn_task == 'multi_classification':
            raise NotImplementedError('Multi-classification not implemented')
        elif self.learn_task == 'regression':
            self.stratify = None
        else:
            raise ValueError(f'Unknown learn task: {self.learn_task}')

    def create_split(self) -> tuple:
        """Split in train and test set"""
        if self.n_bootstraps == 1:
            train, test = train_test_split(
                self.frame,
                stratify=self.stratify,
                test_size=self.test_frac,
                random_state=self.seed,
            )
        else:
            train = resample(
                self.frame,
                replace=True,
                n_samples=(1 - self.test_frac) * len(self.frame.index),
                stratify=self.stratify,
                random_state=self.boot_seed,
            )
            test = self.frame.drop(train.index, axis=0)

        return train, test
