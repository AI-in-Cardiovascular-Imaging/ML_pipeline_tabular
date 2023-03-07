from loguru import logger

from cardio_parsers.data_borg.data_borg import DataBorg


class TargetStatistics(DataBorg):
    def __init__(self, confing):
        super().__init__()
        original_data = self.get_original_data()
        target = original_data[target_label]
        if target.nunique() == 2:  # binary target -> classification
            ratio = (target.sum() / len(target.index)).round(2)
            logger.info(
                f'\nSummary statistics for binary target variable {target_label}:\n'
                f'Positive class makes up {target.sum()} samples out of {len(target.index)}, i.e. {ratio * 100}%.'
            )
            return 'classification', target  # stratify w.r.t. target classes
        else:  # continous target -> regression
            logger.info(
                f'\nSummary statistics for continuous target variable {target_label}:\n'
                f'{target.describe(percentiles=[]).round(2)}'
            )
            return 'regression', None  # do not stratify for regression task
