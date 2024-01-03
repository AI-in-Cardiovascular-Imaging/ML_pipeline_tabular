import os

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
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

from pipeline_tabular.utils.data_split import DataSplit
from pipeline_tabular.utils.helpers import job_name_cleaner
from pipeline_tabular.utils.imputers import Imputer
from pipeline_tabular.utils.normalisers import Normalisers
from pipeline_tabular.utils.selections import Selection
from pipeline_tabular.utils.verifications import Verification
from pipeline_tabular.data_handler.data_handler import DataHandler
from pipeline_tabular.run.report import Report


class Run(DataHandler, Normalisers):
    """Class to run the desired models for multiple seeds/bootstraps as defined in config file"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.experiment_name = config.meta.name
        self.out_dir = config.meta.output_dir
        self.learn_task = config.meta.learn_task
        self.init_seed = config.data_split.init_seed
        self.n_seeds = config.data_split.n_seeds
        self.n_bootstraps = config.data_split.n_bootstraps
        self.oversample = config.data_split.oversample
        self.oversample_method = config.data_split.oversample_method
        self.jobs = config.selection.jobs
        self.config.plot_first_iter = False

        self.imputation = Imputer(self.config)
        self.data_split = DataSplit(self.config)
        self.selection = Selection(self.config)
        self.verification = Verification(self.config)

    def __call__(self) -> None:
        """Iterate over all desired seeds/bootstraps, etc."""
        high_logging_level = self.config.meta.logging_level in ['TRACE', 'DEBUG', 'INFO']
        np.random.seed(self.init_seed)
        seeds = np.random.randint(low=0, high=2**32, size=self.n_seeds)  # generate desired number of random seeds
        report = Report(self.config, seeds)

        if not self.config.meta.collect_results:
            for seed_iter, seed in enumerate(tqdm(seeds, desc='Running seeds', disable=high_logging_level)):
                logger.info(f'Running seed {seed_iter+1}/{self.n_seeds}...')
                np.random.seed(seed)
                boot_seeds = np.random.randint(low=0, high=2**32, size=self.n_bootstraps)  # generate boot seeds
                for boot_iter in range(self.n_bootstraps):
                    logger.info(f'Running bootstrap iteration {boot_iter+1}/{self.n_bootstraps}...')
                    self.data_split(seed, boot_seeds[boot_iter])
                    fit_imputer = self.imputation(seed)
                    if self.oversample:
                        train = self.over_sampling(self.get_store('frame', seed, 'train'), seed)
                        self.set_store('frame', seed, 'train', train)
                    job_names = job_name_cleaner(self.jobs)
                    for job, job_name in zip(self.jobs, job_names):
                        logger.info(f'Running {job_name}...')
                        job_dir = os.path.join(self.out_dir, self.experiment_name, job_name)
                        os.makedirs(job_dir, exist_ok=True)
                        try:
                            features = self.get_store('feature', seed, job_name, boot_iter=boot_iter)
                        except KeyError:
                            features = []
                        if not features:
                            self.selection(
                                seed, boot_iter, job, job_name, job_dir
                            )  # run only if selection results not already available
                        else:
                            norm = [step for step in self.jobs[0] if 'norm' in step][
                                0
                            ]  # need to init normalisation for verification (normally part of selection)
                            train_frame = self.get_store('frame', seed, 'train')
                            _ = getattr(self, norm)(train_frame)
                        _ = self.verification(seed, boot_iter, job_name, job_dir, fit_imputer)

                    self.save_intermediate_results(os.path.join(self.out_dir, self.experiment_name))
                    self.config.plot_first_iter = (
                        False  # minimise work by producing certain plots only for the first iteration
                    )

        report()  # summarise results

    def over_sampling(self, x_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Over sample data"""
        over_sampler_dict = {
            'binary_classification_smoten': SMOTEN(random_state=seed),
            'binary_classification_smotenc': SMOTENC(categorical_features=2, random_state=seed),
            'binary_classification_svmsmote': SVMSMOTE(random_state=seed),
            'binary_classification_borderlinesmote': BorderlineSMOTE(random_state=seed),
            'binary_classification_randomoversampler': RandomOverSampler(random_state=seed),
            'regression_adasyn': ADASYN(random_state=seed),
            'regression_smote': SMOTE(random_state=seed),
            'regression_kmeanssmote': KMeansSMOTE(random_state=seed),
            'regression_randomoversampler': RandomOverSampler(random_state=seed),
        }

        method = self.oversample_method[self.learn_task]
        over_sampler_name = f'{self.learn_task}_{method}'.lower()
        if over_sampler_name not in over_sampler_dict:
            raise ValueError(f'Unknown over sampler: {over_sampler_name}')

        over_sampler = over_sampler_dict[over_sampler_name]
        y_frame = x_frame[self.target_label]
        new_x_frame, _ = over_sampler.fit_resample(x_frame, y_frame)
        return new_x_frame
