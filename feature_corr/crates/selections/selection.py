import os

from crates.normalisers import Normalisers
from loguru import logger
from omegaconf import DictConfig

from feature_corr.crates.selections.dim_reductions import DimensionReductions
from feature_corr.crates.selections.feature_reductions import FeatureReductions
from feature_corr.crates.selections.recursive_feature_elimination import (
    RecursiveFeatureElimination,
)
from feature_corr.data_borg import DataBorg


class Selection(DataBorg, Normalisers, DimensionReductions, FeatureReductions, RecursiveFeatureElimination):
    """Execute jobs"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        self.seed = config.meta.seed
        self.jobs = config.selection.jobs
        self.task = config.meta.learn_task
        self.out_dir = config.meta.output_dir
        self.scoring = config.selection.scoring
        self.experiment_name = config.meta.name
        self.state_name = config.meta.state_name
        self.target_label = config.meta.target_label
        self.corr_method = config.selection.corr_method
        self.corr_thresh = config.selection.corr_thresh
        self.class_weight = config.selection.class_weight
        self.variance_thresh = config.selection.variance_thresh
        self.auto_norm_method = config.selection.auto_norm_method
        self.keep_top_features = config.selection.keep_top_features
        self.corr_drop_features = config.selection.corr_drop_features
        self.job_name = ''
        self.job_dir = None

    def __call__(self) -> None:
        """Run all jobs"""
        self.__check_jobs()
        self.__check_auto_norm_methods()

        for job in self.jobs:
            logger.info(f'Running {job}')
            self.job_name = '_'.join(job)  # name of current job
            self.job_dir = os.path.join(self.out_dir, self.experiment_name, self.state_name, self.job_name)
            os.makedirs(self.job_dir, exist_ok=True)
            data = self.get_store('frame', self.state_name, 'selection_train')
            for step in job:
                data, error = self.process_job(step, data)
                if error:
                    logger.error(f'Step {step} is invalid')
                    break
            self.__store_features(data)

    def __check_jobs(self) -> None:
        """Check if the given jobs are valid"""
        valid_methods = set([x for x in dir(self) if not x.startswith('_') and x != 'process_job'])
        jobs = set([x for sublist in self.jobs for x in sublist])
        if not jobs.issubset(valid_methods):
            raise ValueError(f'Invalid job, check -> {str(jobs - valid_methods)}')

    def __check_auto_norm_methods(self) -> None:
        """Check if auto_norm_method keys are valid"""
        valid_methods = set([x for x in dir(self) if not x.startswith('_') and x.endswith('norm')])
        selected_methods = set(self.auto_norm_method.values())
        if not selected_methods.issubset(valid_methods):
            raise ValueError(f'Invalid auto norm method, check -> {str(selected_methods - valid_methods)}')

    def __store_features(self, data: tuple) -> None:
        """Store features"""
        if isinstance(data, tuple):
            top_features = data[0]
            logger.info(f'Found features')
            self.set_store('feature', self.state_name, self.job_name, top_features)
        else:
            logger.warning(
                f'No features selected for {self.job_name}, add feature reduction or recursive '
                f'feature elimination to your job definition'
            )

    def process_job(self, step: str, data: dict) -> tuple:
        """Process data according to the given step"""
        if data is None:
            logger.warning(
                f'No data available for step: {step} in {self.job_name}. '
                f'\nThe previous step does not seem to produce any output.'
            )
            return None, True
        data = getattr(self, step)(data)
        return data, False
