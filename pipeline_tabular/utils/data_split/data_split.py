import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from pipeline_tabular.data_handler.data_handler import DataHandler


class DataSplit(DataHandler):
    """Split frame in selection and verification"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.n_bootstraps = config.data_split.n_bootstraps
        self.test_frac = config.data_split.test_frac

        self.stratify = None
        self.frame = self.get_frame()

    def __call__(self, seed, boot_seed, boot_iter):
        """Split data"""
        self.seed = seed
        self.boot_seed = boot_seed
        self.boot_iter = boot_iter
        self.split_frame()

    def split_frame(self) -> None:
        self.set_stratification(self.frame)
        train, test = self.create_split()

        if self.test_frac <= 0.0 or self.test_frac >= 1.0:
            raise ValueError('"test_frac" is invalid, must be float in (0.0, 1.0)')

        self.set_store('frame', self.seed, 'train', train)
        self.set_store('frame', self.seed, 'test', test)

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
        if self.n_bootstraps == 1:  # i.e. no bootstrapping, normal train test split
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
