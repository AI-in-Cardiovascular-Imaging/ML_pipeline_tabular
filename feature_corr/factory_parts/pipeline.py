import os

import numpy as np
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
from feature_corr.crates.normalisers import Normalisers
from feature_corr.crates.selections import Selection
from feature_corr.crates.verifications import Verification
from feature_corr.data_handler.data_handler import DataHandler


def run_when_active(func):
    """Decorator to run pipeline step when active"""

    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        if self.config[func_name]['active']:
            logger.info(f'Running {func_name}...')
            return func(self, *args, **kwargs)

    return wrapper


class Pipeline(DataHandler, Normalisers):
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
        self.oversample = config.data_split.oversample
        self.oversample_method = config.data_split.oversample_method
        self.add_seed(self.state_name)
        self.sync_ephemeral_data_to_data_store(self.state_name, 'ephemeral')

        self.imputer_pipeline = Imputer(self.config)
        self.data_splitter = DataSplit(self.config)
        self.selector = Selection(self.config)
        self.verifier = Verification(self.config)

        self.over_sampler_dict = {
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

    def __call__(self) -> None:
        """Iterate over pipeline steps"""
        np.random.seed(self.seed)
        boot_seeds = np.random.randint(low=0, high=2**32, size=self.n_bootstraps)
        for boot_iter in range(self.n_bootstraps):
            logger.info(f'Running bootstrap iteration {boot_iter+1}/{self.n_bootstraps}...')
            self.data_splitter(boot_seeds[boot_iter], boot_iter)
            imputer = self.impute()
            if self.oversample:
                train = self.over_sampling(self.get_store('frame', self.state_name, 'train'))
                self.set_store('frame', self.state_name, 'train', train)
            norm = [step for step in self.jobs[0] if 'norm' in step][
                0
            ]  # need to init first normalisation for verification
            train_frame = self.get_store('frame', self.state_name, 'train')
            _ = getattr(self, norm)(train_frame)
            self.verification(
                boot_iter, 'all_features', None, imputer
            )  # run only once per data split, not for every job

            job_names = job_name_cleaner(self.jobs)
            for job, job_name in zip(self.jobs, job_names):
                logger.info(f'Running {job_name}...')
                job_dir = os.path.join(self.out_dir, self.experiment_name, job_name, self.state_name)
                os.makedirs(job_dir, exist_ok=True)
                features = self.get_store('feature', self.state_name, job_name, boot_iter=str(boot_iter))
                if not features:
                    self.selection(
                        boot_iter, job, job_name, job_dir
                    )  # run only if selection results not already available
                self.verification(boot_iter, job_name, job_dir, imputer)

            self.save_intermediate_results(os.path.join(self.out_dir, self.experiment_name))
            self.config.plot_first_iter = False  # minimise work by producing certain plots only for the first iteration

    @run_when_active
    def impute(self) -> None:
        """Impute data"""
        self.imputer_pipeline()

    @run_when_active
    def selection(self, boot_iter, job, job_name, job_dir):
        """Explore data"""
        self.selector(boot_iter, job, job_name, job_dir)

    @run_when_active
    def verification(self, boot_iter, job_name, job_dir, imputer) -> None:
        """Verify selected features"""
        self.verifier(boot_iter, job_name, job_dir, imputer)

    def over_sampling(self, x_frame: pd.DataFrame) -> pd.DataFrame:
        """Over sample data"""
        method = self.oversample_method[self.learn_task]
        over_sampler_name = f'{self.learn_task}_{method}'.lower()
        if over_sampler_name not in self.over_sampler_dict:
            raise ValueError(f'Unknown over sampler: {over_sampler_name}')

        over_sampler = self.over_sampler_dict[over_sampler_name]
        y_frame = x_frame[self.target_label]
        new_x_frame, _ = over_sampler.fit_resample(x_frame, y_frame)
        return new_x_frame
