import os

from loguru import logger

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
        self.state_name = config.meta.state_name
        self.out_dir = config.meta.output_dir
        self.jobs = config.selection.jobs
        self.add_state_name(self.state_name)
        self.sync_ephemeral_data_to_data_store(self.state_name, 'ephemeral')

    def __call__(self) -> None:
        """Iterate over pipeline steps"""
        self.data_split()
        imputer = self.impute()
        norm = [step for step in self.jobs[0] if 'norm' in step][0]  # need to init first normalisation for verification
        train_frame = self.get_store('frame', self.state_name, 'selection_train')
        _ = getattr(self, norm)(train_frame)
        self.verification('all_features', None, imputer)  # run only once per data split, not for every job

        job_names = job_name_cleaner(self.jobs)
        for job, job_name in zip(self.jobs, job_names):
            logger.info(f'Running -> {job_name}')
            job_dir = os.path.join(self.out_dir, self.experiment_name, job_name, self.state_name)
            os.makedirs(job_dir, exist_ok=True)
            self.selection(job, job_name, job_dir)
            self.verification(job_name, job_dir, imputer)

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
