"""Data exploration module
"""

import os
from copy import deepcopy

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from excel.analysis.utils.analyse_variables import AnalyseVariables, FeatureReduction
from excel.analysis.utils.dim_reduction import DimensionReductions
from excel.analysis.utils.helpers import variance_threshold
from excel.analysis.utils.normalisers import Normaliser


class ExploreData(Normaliser, DimensionReductions, AnalyseVariables, FeatureReduction):
    def __init__(self, data: pd.DataFrame, config: DictConfig) -> None:
        super().__init__()
        self.original_data = data
        self.out_dir = os.path.join(config.dataset.out_dir, '6_exploration', config.analysis.experiment.name)
        self.jobs = config.analysis.run.jobs
        self.seed = config.analysis.run.seed
        self.variance_thresh = config.analysis.run.variance_thresh
        self.corr_method = config.analysis.run.corr_method
        self.corr_thresh = config.analysis.run.corr_thresh
        self.corr_drop_features = config.analysis.run.corr_drop_features
        self.metadata = config.analysis.experiment.metadata
        self.target_label = config.analysis.experiment.target_label
        self.auto_norm_method = config.analysis.run.auto_norm_method
        self.class_weight = 'balanced'

        self.job_name = ''

    def __call__(self) -> pd.DataFrame:
        """Run all jobs"""
        self.__check_jobs()
        self.__check_auto_norm_methods()
        for job in self.jobs:
            logger.info(f'Running {job}')
            self.job_name = '_'.join(job)  # name of current job
            self.job_dir = os.path.join(self.out_dir, self.job_name)
            os.makedirs(self.job_dir, exist_ok=True)
            data = deepcopy(self.original_data)
            for step in job:
                data, error = self.process_job(step, data)
                if error:
                    logger.error(f'Step {step} is invalid')
                    break
        
        if isinstance(data, tuple): # return features
            return data[1]
        else:
            return data.columns

    def __check_jobs(self) -> None:
        """Check if the given jobs are valid"""
        valid_methods = set([x for x in dir(self) if not x.startswith('_') and x != 'process_job'])
        jobs = set([x for sublist in self.jobs for x in sublist])
        if not jobs.issubset(valid_methods):
            raise ValueError(f'Invalide job, check -> {str(jobs - valid_methods)}')

    def __check_auto_norm_methods(self) -> None:
        """Check if auto_norm_method keys are valid"""
        valid_methods = set([x for x in dir(self) if not x.startswith('_') and x.endswith('norm')])
        selected_methods = set(self.auto_norm_method.values())
        if not selected_methods.issubset(valid_methods):
            raise ValueError(f'Invalid auto norm method, check -> {str(selected_methods - valid_methods)}')

    def process_job(self, step, data):
        """Process data according to the given step"""
        if data is None:
            logger.warning(
                f'No data available for step: {step} in {self.job_name}. '
                f'\nThe previous step does not seem to produce any output.'
            )
            return None, True
        data = getattr(self, step)(data)
        return data, False

    def variance_threshold(self, data):
        """Perform variance threshold based feature selection on the data"""
        data = variance_threshold(
            data=data,
            label=self.target_label,
            thresh=self.variance_thresh,
        )
        return data
