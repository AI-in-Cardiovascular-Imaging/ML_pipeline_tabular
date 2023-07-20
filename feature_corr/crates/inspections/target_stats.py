import os

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from feature_corr.data_borg import DataBorg


def check_learn_task(target_frame: pd.DataFrame) -> str:
    """Check if the target variable is binary, multiclass or continuous."""
    if target_frame.nunique() == 2:
        return 'binary_classification'
    if 2 < target_frame.nunique() <= 10:
        return 'multi_classification'
    return 'regression'


class TargetStatistics(DataBorg):
    """Show target statistics"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_label = config.meta.target_label
        self.out_dir = config.meta.output_dir
        self.corr_method = config.selection.corr_method
        self.ephemeral_frame = self.get_frame('ephemeral')
        self.corr_to_target()

    def corr_to_target(self) -> None:
        y = self.ephemeral_frame[self.target_label]
        x = self.ephemeral_frame.drop(self.target_label, axis=1)
        corr_series = x.corrwith(y, axis=0, method=self.corr_method).round(2)
        corr_df = pd.DataFrame({'correlation_to_target': corr_series.values, 'feature': corr_series.index})
        corr_df = corr_df.sort_values(by='correlation_to_target')
        corr_df.to_csv(
            os.path.join(self.out_dir, f'feature_{self.corr_method}_corr_target_{self.target_label}.txt'),
            sep='\t',
            header=True,
            index=False,
        )

    def show_target_statistics(self) -> None:
        """Show target statistics"""
        if self.target_label not in self.ephemeral_frame:
            raise ValueError(f'Target label {self.target_label} not in data')
        target_frame = self.ephemeral_frame[self.target_label]
        task = check_learn_task(target_frame)
        self.set_target_task()
        self._plot_stats(self.target_label, target_frame, task)

    def set_target_task(self) -> None:
        """Set the learning task"""
        target_frame = self.ephemeral_frame[self.target_label]
        learn_task = check_learn_task(target_frame)
        OmegaConf.update(self.config.meta, 'learn_task', learn_task)

    @staticmethod
    def _plot_stats(target_label, target_frame, task) -> None:
        """Show target statistics"""
        if task == 'binary_classification':
            perc = int((target_frame.sum() / len(target_frame.index)).round(2) * 100)
            logger.info(
                f'\nSummary statistics for binary target variable {target_label}:\n'
                f'Positive cases -> {perc}% or {int(target_frame.sum())}/{len(target_frame.index)} samples.'
            )

        elif task == 'multi_classification':
            raise NotImplementedError('Multi-classification is not implemented yet')

        elif task == 'regression':
            logger.info(
                f'\nSummary statistics for continuous target variable {target_label}:\n'
                f'{target_frame.describe(percentiles=[]).round(2)}'
            )

        else:
            raise ValueError(f'Unknown learn task: {task}')
