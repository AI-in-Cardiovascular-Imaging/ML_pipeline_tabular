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


def init_estimator(estimator_name: str, learn_task: str, seed, scoring, class_weight):
    """Initialise the estimator and cross-validation method"""
    if learn_task == 'binary-classification':
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
            raise NotImplementedError(f'The estimator you requested ({estimator_name}) has not yet been implemented.')
        cross_fold = StratifiedKFold(shuffle=True, random_state=seed)

    elif learn_task == 'multi-classification':
        raise NotImplementedError('Multi-classification is not yet implemented.')

    elif learn_task == 'regression':
        if estimator_name == 'forest':
            estimator = RandomForestRegressor(random_state=seed)
        elif estimator_name == 'extreme_forest':
            estimator = ExtraTreesRegressor(random_state=seed)
        elif estimator_name == 'adaboost':
            estimator = AdaBoostRegressor(random_state=seed)
        elif estimator_name == 'logistic_regression':
            raise ValueError('Logistic regression can only be used for classification.')
        elif estimator_name == 'xgboost':
            estimator = GradientBoostingRegressor(random_state=seed)
        else:
            raise NotImplementedError(f'The estimator you requested ({estimator_name}) has not yet been implemented.')
        cross_fold = KFold(shuffle=True, random_state=seed)

    else:
        raise ValueError(f'The learn task you requested ({learn_task}) is not supported.')

    scoring = scoring[learn_task]
    return estimator, cross_fold, scoring
