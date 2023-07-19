import os

import pandas as pd
from loguru import logger
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

from feature_corr.crates.data_split import DataSplit
from feature_corr.crates.helpers import job_name_cleaner
from feature_corr.crates.imputers import Imputer
from feature_corr.crates.inspections import TargetStatistics
from feature_corr.crates.normalisers import Normalisers
from feature_corr.crates.selections import Selection
from feature_corr.crates.verifications import Verification
from feature_corr.data_borg.data_borg import DataBorg


def run_when_active(func):
    """Decorator to run pipeline step when active"""

    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        if self.config[func_name]['active']:
            logger.info(f'Running -> {func_name}')
            return func(self, *args, **kwargs)

    return wrapper


class Pipeline(DataBorg, Normalisers):
    """Pipeline definition"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.experiment_name = config.meta.name
        self.seed = config.meta.seed
        self.state_name = config.meta.state_name
        self.out_dir = config.meta.output_dir
        self.learn_task = config.meta.learn_task
        self.jobs = config.selection.jobs
        self.n_bootstraps = config.data_split.n_bootstraps
        self.over_sample_selection = config.data_split.over_sample_selection
        self.over_sample_verification = config.data_split.over_sample_verification
        self.over_sample_method = config.data_split.over_sample_method
        self.add_state_name(self.state_name)
        self.sync_ephemeral_data_to_data_store(self.state_name, 'ephemeral')

    def __call__(self) -> None:
        """Iterate over pipeline steps"""
        for boot_iter in range(self.n_bootstraps):
            # potentially need to generate new random seed for each iter?
            self.data_split()
            imputer = self.impute()
            if self.over_sample_selection:
                s_train = self.over_sampling(self.get_store('frame', self.state_name, 'train'))
                self.set_store('frame', self.state_name, 'train', s_train)
            if self.over_sample_verification:
                v_train = self.get_store('frame', self.state_name, 'train')
                v_train_imp = imputer.transform(v_train)
                v_train = pd.DataFrame(v_train_imp, index=v_train.index, columns=v_train.columns)
                v_train = self.over_sampling(v_train)
                self.set_store('frame', self.state_name, 'train', v_train)
            norm = [step for step in self.jobs[0] if 'norm' in step][
                0
            ]  # need to init first normalisation for verification
            train_frame = self.get_store('frame', self.state_name, 'train')
            _ = getattr(self, norm)(train_frame)
            self.verification('all_features', None, imputer)  # run only once per data split, not for every job

            job_names = job_name_cleaner(self.jobs)
            for job, job_name in zip(self.jobs, job_names):
                logger.info(f'Running -> {job_name}')
                job_dir = os.path.join(self.out_dir, self.experiment_name, job_name, self.state_name)
                os.makedirs(job_dir, exist_ok=True)
                self.selection(job, job_name, job_dir)
                self.verification(job_name, job_dir, imputer)

            self.config.plot_first_iter = False  # minimise work by producing certain plots only for the first iteration

    def __del__(self):
        """Delete assigned state data store"""
        self.remove_state_data_store(self.state_name)

    def inspection(self) -> None:
        """Inspect data"""
        TargetStatistics(self.config).set_target_task()

    @run_when_active
    def impute(self) -> None:
        """Impute data"""
        imputer = Imputer(self.config)()
        return imputer

    def data_split(self) -> None:
        """Split data"""
        DataSplit(self.config)()

    @run_when_active
    def selection(self, job, job_name, job_dir):
        """Explore data"""
        Selection(self.config)(job, job_name, job_dir)

    @run_when_active
    def verification(self, job_name, job_dir, imputer) -> None:
        """Verify selected features"""
        Verification(self.config)(job_name, job_dir, imputer)

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
