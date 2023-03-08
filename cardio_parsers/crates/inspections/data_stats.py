from loguru import logger

from cardio_parsers.data_borg.data_borg import DataBorg


class TargetStatistics(DataBorg):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_label = config.meta.target_label
        self.original_data = self.get_original_data()

    def show_target_statistics(self) -> None:
        """Show target statistics"""
        for target_label in self.target_label:
            target_data = self.original_data[target_label]
            task = self._check_learning_task(target_data)
            self._plot_stats(target_label, target_data, task)

    @staticmethod
    def _plot_stats(target_label, target_data, task) -> None:
        """Show target statistics"""
        if task == 'binary-classification':
            ratio = (target_data.sum() / len(target_data.index)).round(2)
            logger.info(
                f'\nSummary statistics for binary target variable {target_label}:\n'
                f'Positive class makes up {target_data.sum()} samples out of '
                f'{len(target_data.index)}, i.e. {ratio * 100}%.'
            )

        elif task == 'multi-classification':
            raise NotImplementedError('Multi-classification is not implemented yet')

        elif task == 'regression':
            logger.info(
                f'\nSummary statistics for continuous target variable {target_label}:\n'
                f'{target_data.describe(percentiles=[]).round(2)}'
            )
        else:
            raise ValueError(f'Unknown learning task: {task}')

    @staticmethod
    def _check_learning_task(target_data) -> str:
        """Check if the target variable is binary, multiclass or continuous."""
        if target_data.nunique() == 2:
            return 'binary-classification'
        if 2 < target_data.nunique() <= 10:
            return 'multi-classification'
        return 'regression'

    def get_learning_task(self, target_label) -> str:
        """Get learning task"""
        target_data = self.original_data[target_label]
        return self._check_learning_task(target_data)


# len(target_data.nunique())
# class TargetStatistics(DataBorg):
#
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.target_label = config.meta.target_label
#         self.original_data = self.get_original_data()
#
#     def __call__(self):
#         target_data = self.original_data[self.target_label]
#         if len(target_data.nunique()) == 2:  # binary target -> classification
#             ratio = (target_data.sum() / len(target_data.index)).round(2)
#             logger.info(
#                 f'\nSummary statistics for binary target variable {self.target_label}:\n'
#                 f'Positive class makes up {target_data.sum()} samples out of {len(target_data.index)}, i.e. {ratio * 100}%.'
#             )
#             return 'classification', target_data  # stratify w.r.t. target classes
#         else:  # continous target -> regression
#             logger.info(
#                 f'\nSummary statistics for continuous target variable {self.target_label}:\n'
#                 f'{target_data.describe(percentiles=[]).round(2)}'
#             )
#             return 'regression', None  # do not stratify for regression task
