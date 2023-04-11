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
from sklearn.model_selection import KFold, StratifiedKFold


def init_estimator(
    estimator_name: str, learn_task: str, seed: int, scoring: dict, class_weight: str = None, workers: int = 8
):
    """Initialise the estimator and cross-validation method"""
    estimator_name = f'{estimator_name}_{learn_task}'
    estimator_dict = {
        'forest_binary_classification': RandomForestClassifier(
            random_state=seed, class_weight=class_weight, n_jobs=workers
        ),
        'extreme_forest_binary_classification': ExtraTreesClassifier(
            random_state=seed, class_weight=class_weight, n_jobs=workers
        ),
        'adaboost_binary_classification': AdaBoostClassifier(random_state=seed),
        'logistic_regression_binary_classification': LogisticRegression(
            random_state=seed, class_weight=class_weight, n_jobs=workers
        ),
        'xgboost_binary_classification': GradientBoostingClassifier(random_state=seed),
        'forest_regression': RandomForestRegressor(random_state=seed, n_jobs=workers),
        'extreme_forest_regression': ExtraTreesRegressor(random_state=seed, n_jobs=workers),
        'adaboost_regression': AdaBoostRegressor(random_state=seed),
        'xgboost_regression': GradientBoostingRegressor(random_state=seed),
    }

    if learn_task == 'binary_classification':
        cross_fold = StratifiedKFold(shuffle=True, random_state=seed)
    elif learn_task == 'multi_classification':
        raise NotImplementedError('Multi-classification not implemented')
    elif learn_task == 'regression':
        cross_fold = KFold(shuffle=True, random_state=seed)
    else:
        raise ValueError(f'Unknown learn task: {learn_task}')

    if estimator_name not in estimator_dict:
        raise ValueError(f'Unknown estimator: {estimator_name}')

    estimator = estimator_dict[estimator_name]
    scoring = scoring[learn_task]
    return estimator, cross_fold, scoring


def job_name_cleaner(jobs: list) -> str:
    """Transform jobs given in list into job name strings"""
    job_names = []
    for job in jobs:
        if 'set_memory' in job:
            index = job.index('set_memory')
            steps_before_set = job[:index]  # store steps before set_memory for upcoming jobs
            tmp = [step for step in job if step != 'set_memory']
        elif 'get_memory' in job:
            index = job.index('get_memory')
            tmp = job[index + 1 :]  # keep only steps after get_memory
            tmp = steps_before_set + tmp
        else:
            tmp = job
        job_names.append('_'.join(tmp))

    return job_names
