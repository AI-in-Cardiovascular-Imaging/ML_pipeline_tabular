import os
from copy import deepcopy
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from loguru import logger
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression


def target_statistics(data: pd.DataFrame, target_label: str):
    logger.info('Calculating target statistics...')
    target = data[target_label]
    if target.nunique() == 2:  # binary target
        ratio = (target.sum() / len(target.index)).round(2)
        logger.info(
            f'\nSummary statistics for binary target variable {target_label}:\n'
            f'Positive class makes up {target.sum()} samples out of {len(target.index)}, i.e. {ratio*100}%.'
        )
    else:
        logger.info(
            f'\nSummary statistics for continuous target variable {target_label}:\n'
            f'{target.describe(percentiles=[]).round(2)}'
        )


def save_tables(out_dir, experiment_name, tables) -> None:
    """Save tables to excel file"""
    file_path = os.path.join(out_dir, f'{experiment_name}.xlsx')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tables.to_excel(file_path, index=False)


def split_data(data: pd.DataFrame, metadata: list, hue: str, remove_mdata: bool = True):
    """Split data into data to analyse and hue data"""
    to_analyse = deepcopy(data)
    hue_df = to_analyse[hue]
    if remove_mdata:
        to_analyse = to_analyse.drop(metadata, axis=1, errors='ignore')
    suffix = 'no_mdata' if remove_mdata else 'with_mdata'
    return to_analyse, hue_df, suffix


def variance_threshold(data: pd.DataFrame, label: str, thresh: float) -> pd.DataFrame:
    """Remove features with variance below threshold"""
    tmp = data[label]  # save label col
    selector = VarianceThreshold(threshold=thresh * (1 - thresh))
    selector.fit(data)
    data = data.loc[:, selector.get_support()]
    logger.info(
        f'Removed {len(selector.get_support()) - len(data.columns)} features with same value in more than {int(thresh*100)}% of subjects, '
        f'number of remaining features: {len(data.columns)}'
    )

    if label not in data.columns:  # ensure label col is kept
        logger.warning(f'Target label {label} has variance below threshold {thresh}.')
        data = pd.concat((data, tmp), axis=1)

    return data


def init_estimator(estimator_name: str, classification: bool, seed, class_weight):
    if classification:
        if estimator_name == 'forest':
            estimator = RandomForestClassifier(random_state=seed, class_weight=class_weight)
        elif estimator_name == 'extreme_forest':
            estimator = ExtraTreesClassifier(random_state=seed, class_weight=class_weight)
        elif estimator_name == 'adaboost':
            estimator = AdaBoostClassifier(random_state=seed)
        elif estimator_name == 'logistic_regression':
            estimator = LogisticRegression(random_state=seed, class_weight=class_weight)
        elif estimator_name == 'xgboost':
            estimator = GradientBoostingClassifier(random_state=seed)
        else:
            logger.error(f'The estimator you requested ({estimator_name}) has not yet been implemented.')
            raise NotImplementedError

    else:  # regression
        if estimator_name == 'forest':
            estimator = RandomForestRegressor(random_state=seed, class_weight=class_weight)
        elif estimator_name == 'extreme_forest':
            estimator = ExtraTreesRegressor(random_state=seed, class_weight=class_weight)
        elif estimator_name == 'adaboost':
            estimator = AdaBoostRegressor(random_state=seed)
        elif estimator_name == 'logistic_regression':
            estimator = LogisticRegression(random_state=seed, class_weight=class_weight)
        elif estimator_name == 'xgboost':
            estimator = GradientBoostingRegressor(random_state=seed)
        else:
            logger.error(f'The estimator you requested ({estimator_name}) has not yet been implemented.')
            raise NotImplementedError

    return estimator
