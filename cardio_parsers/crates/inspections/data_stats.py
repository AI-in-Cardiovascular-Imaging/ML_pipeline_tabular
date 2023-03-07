from loguru import logger

from cardio_parsers.data_borg import DataBorg


class TargetStatistics(DataBorg):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.target_label = config.meta.target_label
        print(self.__original_data)

    def __call__(self):
        target_data = self.original_data[self.target_label]
        if target_data.nunique() == 2:  # binary target -> classification
            ratio = (target_data.sum() / len(target_data.index)).round(2)
            logger.info(
                f'\nSummary statistics for binary target variable {self.target_label}:\n'
                f'Positive class makes up {target_data.sum()} samples out of {len(target_data.index)}, i.e. {ratio * 100}%.'
            )
            return 'classification', target_data  # stratify w.r.t. target classes
        else:  # continous target -> regression
            logger.info(
                f'\nSummary statistics for continuous target variable {self.target_label}:\n'
                f'{target_data.describe(percentiles=[]).round(2)}'
            )
            return 'regression', None  # do not stratify for regression task
