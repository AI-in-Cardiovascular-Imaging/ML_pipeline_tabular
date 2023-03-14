import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crates.helpers import init_estimator
from loguru import logger
from sklearn.feature_selection import RFECV


class RecursiveFeatureElimination:
    """Recursive feature elimination"""

    def __init__(self) -> None:
        self.job_dir = None
        self.metadata = None
        self.seed = None
        self.target_label = None
        self.corr_method = None
        self.corr_thresh = None
        self.corr_drop_features = None
        self.scoring = None
        self.class_weight = None
        self.learn_task = None
        self.keep_top_features = None
        np.random.seed(self.seed)

    def __reduction(self, frame: pd.DataFrame, rfe_estimator: str) -> (pd.DataFrame, pd.DataFrame):
        """Reduce the number of features using recursive feature elimination"""
        estimator, cross_validator, scoring = init_estimator(
            rfe_estimator,
            self.learn_task,
            self.seed,
            self.scoring,
            self.class_weight,
        )

        x = frame.drop(self.target_label, axis=1)
        y = frame[self.target_label]
        min_features = 1

        selector = RFECV(
            estimator=estimator,
            min_features_to_select=min_features,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=self.config.meta.workers,
        )
        selector.fit(x, y)

        # Plot performance for increasing number of features
        n_scores = len(selector.cv_results_['mean_test_score'])
        fig = plt.figure()
        plt.xlabel('Number of features selected')
        plt.ylabel(f'Mean {self.scoring}')
        plt.xticks(range(0, n_scores + 1, 5))
        plt.grid(alpha=0.5)
        plt.errorbar(
            range(min_features, n_scores + min_features),
            selector.cv_results_['mean_test_score'],
            yerr=selector.cv_results_['std_test_score'],
        )
        plt.title(f'Recursive Feature Elimination for {rfe_estimator} estimator')
        plt.savefig(os.path.join(self.job_dir, f'RFECV_{rfe_estimator}.pdf'))
        plt.close(fig)

        frame = pd.concat((x.loc[:, selector.support_], frame[self.target_label]), axis=1)  # concat with target label

        try:  # some estimators return feature_importances_ attribute, others coef_
            importances = selector.estimator_.feature_importances_
        except AttributeError:
            logger.warning('Note that absolute coefficient values do not necessarily represent feature importances.')
            importances = np.abs(np.squeeze(selector.estimator_.coef_))

        importances = pd.DataFrame(importances, index=x.columns[selector.support_], columns=['importance'])
        importances = importances.sort_values(by='importance', ascending=True)
        importances = importances.iloc[-self.keep_top_features :, :]  # keep only the top features

        logger.info(
            f'Removed {len(x.columns) + 1 - len(frame.columns)} features with RFE and {rfe_estimator} estimator, '
            f'number of remaining features: {len(frame.columns) - 1}'
        )

        ax = importances.plot.barh()
        fig = ax.get_figure()
        plt.title(
            f'Feature importance (top {self.keep_top_features})'
            f'\n{rfe_estimator} estimator for target: {self.target_label}'
        )
        plt.tight_layout()
        plt.gca().legend_.remove()
        plt.savefig(os.path.join(self.job_dir, f'feature_importance_{rfe_estimator}.pdf'), dpi=fig.dpi)
        plt.close(fig)

        return frame, importances.index.tolist()[::-1]

    def fr_logistic_regression(self, frame: pd.DataFrame) -> tuple:
        """Feature reduction using logistic regression estimator"""
        frame, features = self.__reduction(frame, 'logistic_regression')
        return frame, features

    def fr_forest(self, frame: pd.DataFrame) -> tuple:
        """Feature reduction using random forest estimator"""
        frame, features = self.__reduction(frame, 'forest')
        return frame, features

    def fr_extreme_forest(self, frame: pd.DataFrame) -> tuple:
        """Feature reduction using extreme forest estimator"""
        frame, features = self.__reduction(frame, 'extreme_forest')
        return frame, features

    def fr_adaboost(self, frame: pd.DataFrame) -> tuple:
        """Feature reduction using adaboost estimator"""
        frame, features = self.__reduction(frame, 'adaboost')
        return frame, features

    def fr_xgboost(self, frame: pd.DataFrame) -> tuple:
        """Feature reduction using xgboost estimator"""
        frame, features = self.__reduction(frame, 'xgboost')
        return frame, features

    def fr_all(self, frame: pd.DataFrame) -> tuple:
        """Feature reduction using all estimators in an ensemble manner"""
        number_of_estimators = 4
        _, f_features = self.__reduction(frame, 'forest')
        _, xg_features = self.__reduction(frame, 'xgboost')
        _, ada_features = self.__reduction(frame, 'adaboost')
        _, ef_features = self.__reduction(frame, 'extreme_forest')

        min_len = min(
            len(f_features),
            len(xg_features),
            len(ada_features),
            len(ef_features),
        )  # take the minimum length
        if min_len >= 10:
            min_len = 10

        sub_f_features = f_features[:min_len]  # take the top features
        sub_xg_features = xg_features[:min_len]
        sub_ada_features = ada_features[:min_len]
        sub_ef_features = ef_features[:min_len]

        f_features = {f: i for i, f in enumerate(sub_f_features[::-1], start=1)}  # reverse the order and assign weights
        xg_features = {f: i for i, f in enumerate(sub_xg_features[::-1], start=1)}
        ada_features = {f: i for i, f in enumerate(sub_ada_features[::-1], start=1)}
        ef_features = {f: i for i, f in enumerate(sub_ef_features[::-1], start=1)}

        all_keys = set(f_features.keys()).union(xg_features.keys()).union(ada_features.keys()).union(ef_features.keys())
        feature_scores = pd.DataFrame(all_keys, columns=['feature'])
        feature_scores['forest'] = feature_scores['feature'].apply(lambda key: f_features.get(key))
        feature_scores['xgboost'] = feature_scores['feature'].apply(lambda key: xg_features.get(key))
        feature_scores['adaboost'] = feature_scores['feature'].apply(lambda key: ada_features.get(key))
        feature_scores['extreme_forest'] = feature_scores['feature'].apply(lambda key: ef_features.get(key))
        feature_scores = feature_scores.fillna(0)  # non-selected features get score of 0
        feature_scores['all'] = feature_scores.iloc[:, 1:].sum(axis=1)
        feature_scores = feature_scores.sort_values(by='all', ascending=True)

        ax = feature_scores.plot(
            x='feature',
            y=['forest', 'xgboost', 'adaboost', 'extreme_forest'],
            kind='barh',
            stacked=True,
            colormap='viridis',
        )

        fig = ax.get_figure()
        fig.legend(loc='lower right', borderaxespad=4.5)
        plt.title(f'Feature importance\nAll estimators for target: {self.target_label}')
        plt.xlabel(f'Summed importance (max {number_of_estimators*min_len})')
        plt.tight_layout()
        plt.gca().legend_.remove()
        plt.savefig(os.path.join(self.job_dir, 'feature_importance_all.pdf'), dpi=fig.dpi)
        plt.close(fig)

        logger.info(f'Top features: {list(feature_scores["feature"])}')
        features_to_keep = list(feature_scores["feature"]) + list(self.target_label)
        frame = frame.drop(columns=[c for c in frame.columns if c not in features_to_keep], axis=1)
        features = feature_scores['feature'].tolist()[::-1]
        return frame, features
