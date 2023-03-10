import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance


class FeatureReductions:
    """Simple feature reduction methods"""

    def __init__(self) -> None:
        self.job_dir = None
        self.metadata = None
        self.seed = None
        self.target_label = None
        self.corr_method = None
        self.corr_thresh = None
        self.corr_drop_features = None
        self.scoring = None
        np.random.seed(self.seed)

    def univariate_analysis(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Perform univariate analysis (box plots and distributions)"""
        raise NotImplementedError('Test me first and integrate params')
        data_long = data.melt(id_vars=[self.target_label])
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

        to_analyse = data.drop(self.target_label, axis=1)

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

    @staticmethod
    def highlight(df: pd.DataFrame, lower_limit: np.array, upper_limit: np.array) -> pd.DataFrame:
        """Highlight outliers in a dataframe"""
        raise NotImplementedError('Test me first and integrate params')
        style_df = pd.DataFrame('', index=df.index, columns=df.columns)
        mask = pd.concat(
            [
                ~df.iloc[:, i].between(lower_limit[i], upper_limit[i], inclusive='neither')
                for i in range(lower_limit.size)
            ],
            axis=1,
        )
        style_df = style_df.mask(mask, 'background-color: red')
        style_df.iloc[:, lower_limit.size :] = ''  # uncolor metadata
        return style_df

    def bivariate_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform bivariate analysis"""

    def correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation between features and optionally drop highly correlated ones"""
        raise NotImplementedError('Test me first and integrate params')
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

    def drop_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in the data, optionally removing or further investigating them"""
        # Split data and metadata
        raise NotImplementedError('Test me first and integrate params')
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

    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in the data, optionally removing or further investigating them"""
        # Split data and metadata
        raise NotImplementedError('Test me first and integrate params')
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
            lambda _: self.highlight(df=high_data, lower_limit=lower_limit, upper_limit=upper_limit), axis=None
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

    def variance_threshold(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features with variance below threshold"""
        tmp = data[self.target_label]  # save label col
        selector = VarianceThreshold(threshold=self.corr_thresh * (1 - self.corr_thresh))
        selector.fit(data)
        data = data.loc[:, selector.get_support()]
        logger.info(
            f'Removed {len(selector.get_support()) - len(data.columns)} '
            f'features with same value in more than {int(self.corr_thresh*100)}% of subjects, '
            f'number of remaining features: {len(data.columns)}'
        )

        if self.target_label not in data.columns:  # ensure label col is kept
            logger.warning(f'Target label {self.target_label} has variance below threshold {self.corr_thresh}.')
            data = pd.concat((data, tmp), axis=1)
        return data
