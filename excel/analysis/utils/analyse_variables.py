import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

from excel.analysis.utils.helpers import init_estimator, split_data


class AnalyseVariables:
    def __init__(self) -> None:
        self.job_dir = None
        self.metadata = None
        self.seed = None
        np.random.seed(self.seed)
        self.target_label = None
        self.corr_method = None
        self.corr_thresh = None
        self.corr_drop_features = None
        self.scoring = None

    def univariate_analysis(self, data: pd.DataFrame):
        """
        Perform univariate analysis (box plots and distributions)
        """
        # split data and metadata but keep hue column
        if self.target_label in self.metadata:
            self.metadata.remove(self.target_label)
        to_analyse, _, _ = split_data(data, self.metadata, self.target_label, remove_mdata=True)

        # box plot for each feature w.r.t. target_label
        data_long = to_analyse.melt(id_vars=[self.target_label])
        sns.boxplot(
            data=data_long,
            x='value',
            y='variable',
            hue=self.target_label,
            orient='h',
            meanline=True,
            showmeans=True,
        )
        plt.axvline(x=0, alpha=0.7, color='grey', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.job_dir, f'box_plot_{self.target_label}.pdf'))
        plt.clf()

        to_analyse = to_analyse.drop(self.target_label, axis=1)  # now remove hue column

        # box plot for each feature
        sns.boxplot(data=to_analyse, orient='h', meanline=True, showmeans=True, whis=1.5)
        plt.axvline(x=0, alpha=0.7, color='grey', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.job_dir, 'box_plot.pdf'))
        plt.clf()

        # plot distribution for each feature
        sns.displot(data=to_analyse, kind='kde')
        plt.tight_layout()
        plt.savefig(os.path.join(self.job_dir, 'dis_plot.pdf'))
        plt.clf()
        return data

    def bivariate_analysis(self, data):
        """
        Perform bivariate analysis
        """
        pass

    def correlation(self, data: pd.DataFrame):
        """
        Compute correlation between features and optionally drop highly correlated ones
        """
        to_analyse = data.drop(self.target_label, axis=1)
        target = data[self.target_label]
        matrix = to_analyse.corr(method=self.corr_method).round(2)

        if self.corr_drop_features:
            # calculate feature importances
            estimator = RandomForestClassifier(random_state=self.seed)
            estimator.fit(to_analyse, target)
            perm_importances = permutation_importance(estimator, to_analyse, target, scoring=self.scoring)
            importances = perm_importances.importances_mean
            importances = pd.Series(importances, index=to_analyse.columns)
            importances = importances.sort_values(ascending=False)

            matrix = matrix.reindex(
                index=importances.index, columns=importances.index
            )  # sort matrix w.r.t. feature importance
            abs_corr = matrix.abs()
            upper_tri = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))

            cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.corr_thresh)]
            to_analyse = to_analyse.drop(cols_to_drop, axis=1)
            logger.info(
                f'Removed {len(cols_to_drop)} redundant features with correlation above {self.corr_thresh}, '
                f'number of remaining features: {len(to_analyse.columns)}'
            )
            matrix = to_analyse.corr(method=self.corr_method).round(2)

        # plot correlation heatmap
        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(matrix, annot=True, xticklabels=True, yticklabels=True, cmap='viridis')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.job_dir, 'corr_plot.pdf'))
        plt.close(fig)

        data = pd.concat((to_analyse, data[self.target_label]), axis=1)

        return data

    def drop_outliers(self, data: pd.DataFrame):
        """Detect outliers in the data, optionally removing or further investigating them

        Args:
            data (pd.DataFrame): data
            whiskers (float, optional): determines reach of the whiskers. Defaults to 1.5 (matplotlib default)
            remove (bool, optional): whether to remove outliers. Defaults to True.
            investigate (bool, optional): whether to investigate outliers. Defaults to False.
        """
        # Split data and metadata
        mdata = data[self.metadata]
        whiskers = 1.5
        to_analyse = data.drop(self.metadata, axis=1, errors='ignore')

        # Calculate quartiles, interquartile range and limits
        q1, q3 = np.percentile(to_analyse, [25, 75], axis=0)
        iqr = q3 - q1
        lower_limit = q1 - whiskers * iqr
        upper_limit = q3 + whiskers * iqr
        # logger.debug(f'\nlower limit: {lower_limit}\nupper limit: {upper_limit}')

        to_analyse = to_analyse.mask(to_analyse.le(lower_limit) | to_analyse.ge(upper_limit))
        to_analyse.to_excel(os.path.join(self.job_dir, 'outliers_removed.xlsx'), index=True)

        # Add metadata again
        data = pd.concat((to_analyse, mdata), axis=1)

        # TODO: deal with removed outliers (e.g. remove patient)
        return data

    def detect_outliers(self, data: pd.DataFrame):
        """Detect outliers in the data, optionally removing or further investigating them

        Args:
            data (pd.DataFrame): data
            whiskers (float, optional): determines reach of the whiskers. Defaults to 1.5 (matplotlib default)
            remove (bool, optional): whether to remove outliers. Defaults to True.
            investigate (bool, optional): whether to investigate outliers. Defaults to False.
        """
        # Split data and metadata
        mdata = data[self.metadata]
        whiskers = 1.5
        to_analyse = data.drop(self.metadata, axis=1, errors='ignore')

        # Calculate quartiles, interquartile range and limits
        q1, q3 = np.percentile(to_analyse, [25, 75], axis=0)
        iqr = q3 - q1
        lower_limit = q1 - whiskers * iqr
        upper_limit = q3 + whiskers * iqr
        # logger.debug(f'\nlower limit: {lower_limit}\nupper limit: {upper_limit}')

        high_data = to_analyse.copy(deep=True)
        # Remove rows without outliers
        # high_data = high_data.drop(high_data.between(lower_limit, upper_limit).all(), axis=0)
        # Add metadata again
        high_data = pd.concat((high_data, mdata), axis=1).sort_values(by=['subject'])

        # Highlight outliers in table
        high_data.style.apply(
            lambda _: highlight(df=high_data, lower_limit=lower_limit, upper_limit=upper_limit), axis=None
        ).to_excel(os.path.join(self.job_dir, 'investigate_outliers.xlsx'), index=True)
        return data

    def feature_wiz(self, data: pd.DataFrame) -> pd.DataFrame:
        """Use feature_wiz to select features"""
        from featurewiz import FeatureWiz

        y_train = data[self.target_label]
        x_train = data.drop(self.target_label, axis=1)

        features = FeatureWiz(
            corr_limit=self.corr_thresh,
            feature_engg='',
            category_encoders='',
            dask_xgboost_flag=False,
            nrows=None,
            verbose=2,
        )
        selected_features = features.fit_transform(x_train, y_train)
        return selected_features.join(y_train)


def highlight(df: pd.DataFrame, lower_limit: np.array, upper_limit: np.array):
    """Highlight outliers in a dataframe"""
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    mask = pd.concat(
        [~df.iloc[:, i].between(lower_limit[i], upper_limit[i], inclusive='neither') for i in range(lower_limit.size)],
        axis=1,
    )
    style_df = style_df.mask(mask, 'background-color: red')
    style_df.iloc[:, lower_limit.size :] = ''  # uncolor metadata
    return style_df


class FeatureReduction:
    def __init__(self) -> None:
        self.job_dir = None
        self.metadata = None
        self.seed = None
        np.random.seed(self.seed)
        self.target_label = None
        self.corr_method = None
        self.corr_thresh = None
        self.corr_drop_features = None
        self.scoring = None
        self.class_weight = None
        self.task = None

    def __reduction(self, data: pd.DataFrame, rfe_estimator: str) -> (pd.DataFrame, pd.DataFrame):
        estimator, cross_validator, scoring = init_estimator(
            rfe_estimator, self.task, self.seed, self.scoring, self.class_weight
        )

        number_of_top_features = 30
        X = data.drop(self.target_label, axis=1)
        y = data[self.target_label]

        min_features = 1

        selector = RFECV(
            estimator=estimator,
            min_features_to_select=min_features,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=4,
        )
        selector.fit(X, y)

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

        data = pd.concat((X.loc[:, selector.support_], data[self.target_label]), axis=1)  # concat with target label

        try:  # some estimators return feature_importances_ attribute, others coef_
            importances = selector.estimator_.feature_importances_
        except AttributeError:
            logger.warning(f'Note that absolute coefficient values do not necessarily represent feature importances.')
            importances = np.abs(np.squeeze(selector.estimator_.coef_))

        importances = pd.DataFrame(importances, index=X.columns[selector.support_], columns=['importance'])
        importances = importances.sort_values(by='importance', ascending=True)
        importances = importances.iloc[
            -number_of_top_features:, :
        ]  # keep only the top 20 features with the highest importance

        logger.info(
            f'Removed {len(X.columns) + 1 - len(data.columns)} features with RFE and {rfe_estimator} estimator, '
            f'number of remaining features: {len(data.columns) - 1}'
        )

        # plot importances as bar chart
        ax = importances.plot.barh()
        fig = ax.get_figure()
        plt.title(
            f'Feature importance (top {number_of_top_features})'
            f'\n{rfe_estimator} estimator for target: {self.target_label}'
        )
        plt.tight_layout()
        plt.gca().legend_.remove()
        plt.savefig(os.path.join(self.job_dir, f'feature_importance_{rfe_estimator}.pdf'), dpi=fig.dpi)
        plt.close(fig)

        return data, importances.index.tolist()[::-1]

    def fr_logistic_regression(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature reduction using logistic regression estimator"""
        data, features = self.__reduction(data, 'logistic_regression')
        return data, features

    def fr_forest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature reduction using random forest estimator"""
        data, features = self.__reduction(data, 'forest')
        return data, features

    def fr_extreme_forest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature reduction using extreme forest estimator"""
        data, features = self.__reduction(data, 'extreme_forest')
        return data, features

    def fr_adaboost(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature reduction using adaboost estimator"""
        data, features = self.__reduction(data, 'adaboost')
        return data, features

    def fr_xgboost(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature reduction using xgboost estimator"""
        data, features = self.__reduction(data, 'xgboost')
        return data, features

    def fr_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature reduction using all estimators in an ensemble manner"""
        number_of_estimators = 4
        _, f_features = self.__reduction(data, 'forest')
        _, xg_features = self.__reduction(data, 'xgboost')
        _, ada_features = self.__reduction(data, 'adaboost')
        _, ef_features = self.__reduction(data, 'extreme_forest')

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
        data = data.drop(columns=[c for c in data.columns if c not in features_to_keep], axis=1)
        return data, feature_scores['feature'].tolist()[::-1]
