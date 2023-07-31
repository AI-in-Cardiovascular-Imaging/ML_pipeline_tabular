import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from feature_corr.crates.normalisers import Normalisers
from feature_corr.crates.selections.dimension_projections import DimensionProjections
from feature_corr.crates.selections.feature_reductions import FeatureReductions
from feature_corr.crates.selections.memoization import Memoize
from feature_corr.crates.selections.recursive_feature_elimination import (
    RecursiveFeatureElimination,
)
from feature_corr.data_borg import DataBorg


class Selection(DataBorg, Normalisers, DimensionProjections, FeatureReductions, RecursiveFeatureElimination, Memoize):
    """Execute jobs"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.plot_format = config.meta.plot_format
        self.seed = config.meta.seed
        self.workers = config.meta.workers
        self.jobs = config.selection.jobs
        self.task = config.meta.learn_task
        self.scoring = config.selection.scoring
        self.univariate_thresh = config.selection.univariate_thresh
        self.state_name = config.meta.state_name
        self.target_label = config.meta.target_label
        self.corr_method = config.selection.corr_method
        self.corr_thresh = config.selection.corr_thresh
        self.corr_ranking = config.selection.corr_ranking
        self.variance_thresh = config.selection.variance_thresh
        self.class_weight = config.selection.class_weight
        self.param_grids = config.verification.param_grids
        self.n_top_features = config.verification.use_n_top_features
        self.job_name = ''
        self.job_dir = None

    def __call__(self, job, job_name, job_dir) -> None:
        """Run all jobs"""
        self.__check_jobs()
        self.job_name = job_name
        self.job_dir = job_dir

        frame = self.get_store('frame', self.state_name, 'train')
        for step in job:
            logger.info(f'Running {step} for state {self.state_name}...')
            frame, features, error = self.process_job(step, frame)
            if error:
                logger.error(f'Step {step} is invalid')
                break
            self.__store_features(features, step)

    def __check_jobs(self) -> None:
        """Check if the given jobs are valid"""
        valid_methods = set([x for x in dir(self) if not x.startswith('_') and x != 'process_job'])
        jobs = set([x for sublist in self.jobs for x in sublist])
        if not jobs.issubset(valid_methods):
            raise ValueError(f'Invalid job, check -> {str(jobs - valid_methods)}')

    def __store_features(self, features: list, step: str) -> None:
        """Store features"""
        if features:
            if isinstance(features, list):
                if len(features) > 0:
                    self.set_store('feature', self.state_name, self.job_name, features)
                else:
                    logger.warning(f'No features found for {self.job_name}')
            else:
                logger.warning('Selected features are not a list')

    def process_job(self, step: str, frame: pd.DataFrame) -> tuple:
        """Process data according to the given step"""
        if frame is None:
            logger.warning(
                f'No data available for step: {step} in {self.job_name}. '
                f'\nThe previous step does not seem to produce any output.'
            )
            return None, None, True
        frame, features = getattr(self, step)(frame)
        return frame, features, False

