import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import RFECV, RFE

from feature_corr.utils.helpers import init_estimator
from feature_corr.utils.verifications.verification import CrossValidation


class RecursiveFeatureElimination:
    """Recursive feature elimination"""

    def __init__(self) -> None:
        self.config = None
        self.job_dir = None
        self.plot_format = None
        self.metadata = None
        self.workers = None
        self.target_label = None
        self.corr_method = None
        self.corr_thresh = None
        self.corr_drop_features = None
        self.scoring = None
        self.class_weight = None
        self.learn_task = None
        self.param_grids = None

    def __reduction(self, frame: pd.DataFrame, rfe_estimator: str, seed: int) -> tuple:
        """Reduce the number of features using recursive feature elimination"""
        estimator, cross_validator, scoring = init_estimator(
            rfe_estimator, self.learn_task, seed, self.scoring, self.class_weight, self.workers
        )

        y = frame[self.target_label]
        x = frame.drop(self.target_label, axis=1)
        min_features = 1
        optimiser = CrossValidation(
            x,
            y,
            estimator,
            cross_validator,
            self.param_grids[rfe_estimator],
            scoring,
            seed,
            self.workers,
        )
        estimator = optimiser()  # find estimator with ideal parameters

        selector = RFECV(
            estimator=estimator.best_estimator_,
            min_features_to_select=min_features,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=self.workers,
        )
        selector.fit(x, y)

        # Plot performance for increasing number of features
        if self.config.plot_first_iter:
            n_scores = len(selector.cv_results_['mean_test_score'])
            fig = plt.figure()
            plt.xlabel('Number of features selected')
            plt.ylabel(f'Mean {scoring}')
            plt.xticks(range(0, n_scores + 1, 5))
            plt.grid(alpha=0.5)
            plt.errorbar(
                range(min_features, n_scores + min_features),
                selector.cv_results_['mean_test_score'],
                yerr=selector.cv_results_['std_test_score'],
            )
            plt.title(f'Recursive Feature Elimination for {rfe_estimator} estimator')
            plt.savefig(os.path.join(self.job_dir, f'RFECV_{rfe_estimator}.{self.plot_format}'))
            plt.close(fig)

        frame = pd.concat((x.loc[:, selector.support_], frame[self.target_label]), axis=1)  # concat with target label

        try:  # some estimators return feature_importances_ attribute, others coef_
            importances = selector.estimator_.feature_importances_
        except AttributeError:
            logger.warning('Note that absolute coefficient values do not necessarily represent feature importances.')
            importances = np.abs(np.squeeze(selector.estimator_.coef_))

        importances = pd.DataFrame(importances, index=x.columns[selector.support_], columns=['importance'])
        importances = importances.sort_values(by='importance', ascending=True)

        logger.info(
            f'Removed {len(x.columns) + 1 - len(frame.columns)} features with RFE and {rfe_estimator} estimator, '
            f'number of remaining features: {len(frame.columns) - 1}'
        )
        if self.config.plot_first_iter:
            ax = importances.plot.barh()
            fig = ax.get_figure()
            plt.title(f'Feature importance' f'\n{rfe_estimator} estimator for target: {self.target_label}')
            plt.xlabel('Feature importance')
            plt.tight_layout()
            plt.gca().legend_.remove()
            plt.savefig(os.path.join(self.job_dir, f'feature_importance_{rfe_estimator}.{self.plot_format}'), dpi=fig.dpi)
            plt.close(fig)

        features = importances.index.tolist()[::-1]

        return frame, features

    def fr_logistic_regression(self, frame: pd.DataFrame, seed: int) -> tuple:
        """Feature reduction using logistic regression estimator"""
        frame, features = self.__reduction(frame, 'logistic_regression', seed)
        return frame, features

    def fr_forest(self, frame: pd.DataFrame, seed: int) -> tuple:
        """Feature reduction using random forest estimator"""
        frame, features = self.__reduction(frame, 'forest', seed)
        return frame, features

    def fr_extreme_forest(self, frame: pd.DataFrame, seed: int) -> tuple:
        """Feature reduction using extreme forest estimator"""
        frame, features = self.__reduction(frame, 'extreme_forest', seed)
        return frame, features

    def fr_adaboost(self, frame: pd.DataFrame, seed: int) -> tuple:
        """Feature reduction using adaboost estimator"""
        frame, features = self.__reduction(frame, 'adaboost', seed)
        return frame, features

    def fr_xgboost(self, frame: pd.DataFrame, seed: int) -> tuple:
        """Feature reduction using xgboost estimator"""
        frame, features = self.__reduction(frame, 'xgboost', seed)
        return frame, features