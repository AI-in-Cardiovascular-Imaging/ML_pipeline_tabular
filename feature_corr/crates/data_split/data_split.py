import pandas as pd
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

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
        self.selection_frac = config.data_split.selection_frac
        self.over_sample_method = config.data_split.over_sample_method
        self.over_sample_selection = config.data_split.over_sample_selection
        self.verification_test_frac = config.data_split.verification_test_frac
        self.over_sample_verification = config.data_split.over_sample_verification

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
        if self.frame.isnull().values.any():
            raise ValueError('Data contains NaN values, clean up or impute data first')

        self.set_stratification(self.frame)
        s_frame, v_frame = self.create_selection_verification_set()

        if self.verification_test_frac <= 0.0 or self.verification_test_frac >= 1.0:
            raise ValueError('"verification_test_frac" is invalid, must be float between (0.0, 1.0)')

        if self.selection_frac < 1.0:
            v_train, v_test = self.create_verification_split(v_frame)
            s_train = s_frame
        else:
            s_train, v_train, v_test = s_frame, s_frame, v_frame

        # self.show_stats(s_train, v_train, v_test, 'Data split stats')

        if self.over_sample_selection:
            s_train = self.over_sampling(s_train)
        if self.over_sample_verification:
            v_train = self.over_sampling(v_train)

        # if self.over_sample_selection or self.over_sample_verification:
        #     self.show_stats(s_train, v_train, v_test, 'Data split stats after over sampling')

        self.set_store('frame', self.state_name, 'selection_train', s_train)
        self.set_store('frame', self.state_name, 'verification_train', v_train)
        self.set_store('frame', self.state_name, 'verification_test', v_test)
        all_features = list(s_train.columns.drop(self.target_label))
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
        if 0.0 < self.selection_frac < 1.0:
            test_size = 1.0 - self.selection_frac
        elif self.selection_frac == 1.0:  # entire train data is used for selection and verification
            test_size = self.verification_test_frac
        else:
            raise ValueError('Invalid "selection_frac" must be in range -> 0.0 < x <= 1.0')

        s_frame, v_frame = train_test_split(
            self.frame,
            stratify=self.stratify,
            test_size=test_size,
            random_state=self.seed,
        )
        return s_frame, v_frame

    def create_verification_split(self, v_frame: pd.DataFrame) -> tuple:
        """Create verification split if needed"""
        self.set_stratification(v_frame)
        v_train, v_test = train_test_split(
            v_frame,
            stratify=self.stratify,
            test_size=self.verification_test_frac,
            random_state=self.seed,
        )
        return v_train, v_test

    def over_sampling(self, x_frame: pd.DataFrame) -> pd.DataFrame:
        """Over sample data"""
        method = self.over_sample_method[self.learn_task]
        over_sampler_name = f'{self.learn_task}_{method}'.lower()
        over_sampler_dict = {
            'binary_classification_smoten': SMOTEN(random_state=self.seed),
            'binary_classification_smotenc': SMOTENC(categorical_features=2, random_state=self.seed),
            'binary_classification_svmsmote': SVMSMOTE(random_state=self.seed),
            'binary_classification_borderlinesmote': BorderlineSMOTE(random_state=self.seed),
            'binary_classification_randomoversampler': RandomOverSampler(random_state=self.seed),
            'regression_adasyn': ADASYN(random_state=self.seed),
            'regression_smote': SMOTE(random_state=self.seed),
            'regression_kmeanssmote': KMeansSMOTE(random_state=self.seed),
            'regression_randomoversampler': RandomOverSampler(random_state=self.seed),
        }

        if over_sampler_name not in over_sampler_dict:
            raise ValueError(f'Unknown over sampler: {over_sampler_name}')

        over_sampler = over_sampler_dict[over_sampler_name]
        y_frame = x_frame[self.target_label]
        new_x_frame, _ = over_sampler.fit_resample(x_frame, y_frame)
        return new_x_frame
