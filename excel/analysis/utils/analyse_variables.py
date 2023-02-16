import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from numpy import sort
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

from excel.analysis.utils.helpers import split_data


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
        self.rfe_estimator = None

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
        matrix = to_analyse.corr(method=self.corr_method).round(2)

        if self.corr_drop_features:
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
        plt.figure(figsize=(20, 20))
        sns.heatmap(matrix, annot=True, xticklabels=True, yticklabels=True, cmap='viridis')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.job_dir, 'corr_plot.pdf'))
        plt.clf()

        data = pd.concat((to_analyse, data[self.target_label]), axis=1)

        return data

    def feature_reduction(self, data: pd.DataFrame):
        """
        Calculate feature importance and remove features with low importance
        """
        if self.rfe_estimator == 'forest':
            estimator = RandomForestClassifier(random_state=self.seed)
        elif self.rfe_estimator == 'extreme_forest':
            estimator = ExtraTreesClassifier(random_state=self.seed)
        elif self.rfe_estimator == 'adaboost':
            estimator = AdaBoostClassifier(random_state=self.seed)
        elif self.rfe_estimator == 'logistic_regression':
            estimator = LogisticRegression(random_state=self.seed)
        else:
            logger.error(f'The RFE estimator you requested ({self.rfe_estimator}) has not yet been implemented.')
            raise NotImplementedError

        X = data.drop(self.target_label, axis=1)
        y = data[self.target_label]

        min_features = 1
        cross_validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        selector = RFECV(
            estimator=estimator,
            step=1,
            min_features_to_select=min_features,
            cv=cross_validator,
            scoring='average_precision',
            n_jobs=4,
        )
        selector.fit(X, y)

        # Plot performance for increasing number of features
        n_scores = len(selector.cv_results_['mean_test_score'])
        plt.figure()
        plt.xlabel('Number of features selected')
        plt.ylabel('Mean average precision')
        plt.xticks(range(0, n_scores + 1, 5))
        plt.grid(alpha=0.5)
        plt.errorbar(
            range(min_features, n_scores + min_features),
            selector.cv_results_['mean_test_score'],
            yerr=selector.cv_results_['std_test_score'],
        )
        plt.title(f'Recursive Feature Elimination for {self.rfe_estimator} estimator')
        plt.savefig(os.path.join(self.job_dir, f'RFECV_{self.rfe_estimator}.pdf'))
        plt.clf()

        data = pd.concat((X.loc[:, selector.support_], data[self.target_label]), axis=1)

        try: # some estimators return feature_importances_ attribute, others coef_
            importances = selector.estimator_.feature_importances_
        except AttributeError:
            logger.warning(f'Note that absolute coefficient values do not necessarily represent feature importances.')
            importances = np.abs(np.squeeze(selector.estimator_.coef_))

        importances = pd.Series(importances, index=X.columns[selector.support_])
        importances = importances.sort_values(ascending=True)
        logger.info(
            f'Removed {len(X.columns) + 1 - len(data.columns)} features with RFE and {self.rfe_estimator} estimator, '
            f'number of remaining features: {len(data.columns) - 1}'
        )

        # Plot importances
        fig = plt.figure(figsize=(10, 10))
        importances.plot.barh()
        plt.title(f'Feature importances using {self.rfe_estimator} estimator for target label: {self.target_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.job_dir, f'feature_importance_{self.rfe_estimator}.pdf'))
        plt.clf()

        # Plot patient/feature value heatmap
        # plt.figure(figsize=(figsize, figsize))
        # sns.heatmap(data.transpose(), annot=False, xticklabels=False, yticklabels=True, cmap='viridis')
        # plt.xticks(rotation=90)
        # fig.tight_layout()
        # plt.savefig(os.path.join(self.job_dir, 'heatmap_after_reduction.pdf'))
        # plt.clf()
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

    # TODO: maybe helper function
    def __test_train_data_split(self, data: pd.DataFrame, test_frac: float = 0.1) -> tuple:
        """Split data into training and test set"""
        y_train = data[self.target_label]
        x_train = data.drop(self.target_label, axis=1)
        samples = int(len(x_train) * test_frac)
        row_idx = np.random.random_integers(0, int(len(x_train) - 1), samples)
        x_test = x_train.take(row_idx, axis=0)
        x_train = x_train.drop(x_test.index)
        y_test = y_train.take(row_idx, axis=0)
        y_train = y_train.drop(x_test.index)
        return x_train, x_test, y_train, y_test

    def __run_xgboost(self, data: pd.DataFrame) -> tuple:
        """Run XGBoost model"""
        x_train, x_test, y_train, y_test = self.__test_train_data_split(data)
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        thresholds = sort(model.feature_importances_)
        for thresh in thresholds:
            selection = SelectFromModel(model, threshold=thresh, prefit=True)  # select features using threshold
            select_x_train = selection.transform(x_train.values)
            selection_model = GradientBoostingClassifier(random_state=self.seed)
            selection_model.fit(select_x_train, y_train)  # train model
            logger.trace(f'Thresh={thresh:.3f}, n={select_x_train.shape[1]}, Acc: {model.score(x_test, y_test):.3f}')
        feature_names = x_train.columns
        return model, feature_names

    def xgboost(self, data: pd.DataFrame) -> pd.DataFrame:
        """Use XGBoost to select features"""
        model, feature_names = self.__run_xgboost(data)
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)

        fig = plt.figure(figsize=(10, 10))
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title(f'Feature importances using XGBoost estimator for target label: {self.target_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.job_dir, f'feature_importance_xgboost.pdf'), dpi=fig.dpi)
        plt.clf()
        return data


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
