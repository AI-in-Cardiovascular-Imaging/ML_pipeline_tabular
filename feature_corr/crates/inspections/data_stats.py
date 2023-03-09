import pandas as pd
from loguru import logger

from feature_corr.data_borg import DataBorg


def check_learn_task(target_data: pd.DataFrame) -> str:
    """Check if the target variable is binary, multiclass or continuous."""
    if target_data.nunique() == 2:
        return 'binary_classification'
    if 2 < target_data.nunique() <= 10:
        return 'multi_classification'
    return 'regression'


class TargetStatistics(DataBorg):
    """Show target statistics"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_label = config.meta.target_label
        self.original_data = self.get_frame('original')
        logger.info(f'Running -> {self.__class__.__name__}')

    def show_target_statistics(self) -> None:
        """Show target statistics"""
        for target_label in self.target_label:
            target_data = self.original_data[target_label]
            task = check_learn_task(target_data)
            self._plot_stats(target_label, target_data, task)

    def set_target_task(self) -> str:
        """Set the learning task"""
        target_data = self.original_data[self.target_label]
        task = check_learn_task(target_data)
        return task

    @staticmethod
    def _plot_stats(target_label, target_data, task) -> None:
        """Show target statistics"""
        if task == 'binary_classification':
            ratio = (target_data.sum() / len(target_data.index)).round(2)
            logger.info(
                f'\nSummary statistics for binary target variable {target_label}:\n'
                f'Positive cases -> {(ratio * 100).round(2)}% or {target_data.sum()}/{len(target_data.index)} samples.'
            )

        elif task == 'multi_classification':
            raise NotImplementedError('Multi-classification is not implemented yet')

        elif task == 'regression':
            logger.info(
                f'\nSummary statistics for continuous target variable {target_label}:\n'
                f'{target_data.describe(percentiles=[]).round(2)}'
            )

        else:
            raise ValueError(f'Unknown learn task: {task}')
