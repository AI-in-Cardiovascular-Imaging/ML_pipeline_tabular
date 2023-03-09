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
from sklearn.model_selection import train_test_split

from feature_corr.data_borg import DataBorg


class DataSplit(DataBorg):
    """Split data in selection and verification"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.seed = config.meta.seed
        self.state_name = config.meta.state_name
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.selection_frac = config.data_split.selection_frac
        self.verification_test_frac = config.data_split.verification_test_frac
        self.stratify = None
        self.frame = self.get_store('frame', self.state_name, 'ephemeral')

    def __call__(self):
        """Split data"""
        self.set_stratification()
        self.create_selection_verification_set()

    def set_stratification(self):
        """Set stratification"""
        if self.learn_task == 'binary-classification':
            target_frame = self.frame[self.config.meta.target_label]
            self.stratify = target_frame
        elif self.learn_task == 'multi-classification':
            raise NotImplementedError('Multi-classification not implemented')
        elif self.learn_task == 'regression':
            self.stratify = None
        else:
            raise ValueError(f'Unknown learn task: {self.learn_task}')

    def create_selection_verification_set(self):
        """Split in selection and verification set"""
        if 0.0 < self.selection_frac < 1.0:
            selection_train, verification_train = train_test_split(
                self.frame,
                stratify=self.stratify,
                test_size=1.0 - self.selection_frac,
                random_state=self.seed,
            )
            self.oversampling(self.frame, self.frame[self.target_label])
            self.set_store('frame', self.state_name, 'selection_train', selection_train)
            self.set_store('frame', self.state_name, 'verification_train', verification_train)
            self.set_store('frame', self.state_name, 'verification_test', None)
            logger.info(
                f'\nData split:'
                f'\nselection_train -> {len(selection_train)}'
                f'\nverification_train -> {len(verification_train)}'
                f'\nverification_test -> None'
            )

        elif (
            self.selection_frac == 0.0
        ):  # special mode in which entire train data is used for selection and verification
            self.oversampling(self.frame, self.frame[self.target_label])
            verification_train, verification_test = train_test_split(
                self.frame,
                stratify=self.stratify,
                test_size=self.verification_test_frac,
                random_state=self.seed,
            )
            self.set_store('frame', self.state_name, 'selection_train', verification_train)
            self.set_store('frame', self.state_name, 'verification_train', verification_train)
            self.set_store('frame', self.state_name, 'verification_test', verification_test)
            logger.info(
                f'\nData split:'
                f'\nselection_train -> {len(verification_train)}'
                f'\nverification_train -> {len(verification_train)}'
                f'\nverification_test -> {len(verification_test)}'
            )
        else:
            raise ValueError(f'Value {self.selection_frac} is invalid, must be float between (0.0, 1.0)')

    def oversampling(self, x_frame, y_frame) -> tuple:
        """Oversample data"""
        oversampler = RandomOverSampler(random_state=self.seed)
        x_frame, y_frame = oversampler.fit_resample(x_frame, y_frame)
        logger.warning(f'asdasd -> {type(x_frame)}')
        # return self.x_train, self.y_train

    # def create_verification_split(self):
    #     """Split verification data in train and test set"""
    #     x, y = self.prepare_data(v_data, features_to_keep=features)
    #     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
    #         x,
    #         y,
    #         stratify=y,
    #         test_size=0.20,
    #         random_state=self.seed,
    #     )

    # def prepare_data(self, data: pd.DataFrame, features_to_keep: list = None) -> tuple:
    #     """Prepare data for verification"""
    #     y = data[self.target_label]
    #     data = self.z_score_norm(data)
    #     x = data.drop(
    #         columns=[c for c in data.columns if c not in features_to_keep], axis=1
    #     )  # Keep only selected features
    #     if self.target_label in x.columns:  # ensure that target column is dropped
    #         x = x.drop(self.target_label, axis=1)
    #     return x, y
