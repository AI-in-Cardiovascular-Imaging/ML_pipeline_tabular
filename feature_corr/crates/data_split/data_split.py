import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from feature_corr.data_borg import DataBorg


class DataSplit(DataBorg):
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

    def __call__(self):
        """Split data"""
        self.split_frame()

    def verification_mode(self, frame: pd.DataFrame) -> None:
        """Split data in selection and verification"""
        tmp_state = self.state_name
        self.state_name = 'verification'
        self.frame = frame
        self.split_frame()
        self.state_name = tmp_state

    def split_frame(self) -> None:
        self.set_stratification(self.frame)
        s_frame, v_frame = self.create_selection_verification_set()

        if self.test_frac <= 0.0 or self.test_frac >= 1.0:
            raise ValueError('"test_frac" is invalid, must be float between (0.0, 1.0)')

        self.set_store('frame', self.state_name, 'train', s_frame)
        self.set_store('frame', self.state_name, 'test', v_frame)
        all_features = list(s_frame.columns.drop(self.target_label))
        self.set_store('feature', self.state_name, 'all_features', all_features)

    def show_stats(self, s_train: pd.DataFrame, v_train: pd.DataFrame, v_test: pd.DataFrame, head: str) -> None:
        """Show data split stats"""
        logger.info(
            f'\n{head}\n'
            f'{"Set":<24}{"rows":<7}{"cols"}\n'
            f'{"Original data:":<24}{len(self.frame):<7}{len(self.frame.columns)}\n'
            f'{"Selection train:":<24}{len(s_train):<7}{len(s_train.columns)}\n'
            f'{"Verification train:":<24}{len(v_train):<7}{len(v_train.columns)}\n'
            f'{"Verification test:":<24}{len(v_test):<7}{len(v_test.columns)}'
        )

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

    def create_selection_verification_set(self) -> tuple:
        """Split in selection and verification set"""
        if self.n_bootstraps == 1:
            s_frame, v_frame = train_test_split(
                self.frame,
                stratify=self.stratify,
                test_size=self.test_frac,
                random_state=self.seed,
            )
        else:
            s_frame = resample(
                self.frame,
                replace=True,
                n_samples=(1 - self.test_frac) * len(self.frame.index),
                stratify=self.stratify,
                random_state=None,
            )
            v_frame = self.frame.drop(s_frame.index, axis=0)

        return s_frame, v_frame
