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
        self.config = None
        self.job_dir = None
        self.metadata = None
        self.seed = None
        self.target_label = None
        self.corr_method = None
        self.corr_thresh = None
        self.corr_drop_features = None
        self.scoring = None
        self.learn_task = None
        np.random.seed(self.seed)

    def univariate_analysis(self, frame: pd.DataFrame) -> tuple:
        """Perform univariate analysis (box plots and distributions)"""
        frame_long = frame.melt(id_vars=[self.target_label])
        sns.boxplot(
            data=frame_long,
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

        x_frame = frame.drop(self.target_label, axis=1)

        # box plot for each feature
        sns.boxplot(data=x_frame, orient='h', meanline=True, showmeans=True, whis=1.5)
        plt.axvline(x=0, alpha=0.7, color='grey', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.job_dir, 'box_plot.pdf'))
        plt.clf()

        # plot distribution for each feature
        sns.displot(data=x_frame, kind='kde')
        plt.tight_layout()
        plt.savefig(os.path.join(self.job_dir, 'dis_plot.pdf'))
        plt.clf()
        return frame, None

    @staticmethod
    def highlight(frame: pd.DataFrame, lower_limit: np.array, upper_limit: np.array) -> tuple:
        """Highlight outliers in a dataframe"""
        style_frame = pd.DataFrame('', index=frame.index, columns=frame.columns)
        mask = pd.concat(
            [
                ~frame.iloc[:, i].between(lower_limit[i], upper_limit[i], inclusive='neither')
                for i in range(lower_limit.size)
            ],
            axis=1,
        )
        style_frame = style_frame.mask(mask, 'background-color: red')
        style_frame.iloc[:, lower_limit.size :] = ''  # uncolor metadata
        return style_frame, None

    def bivariate_analysis(self, frame: pd.DataFrame) -> tuple:
        """Perform bivariate analysis"""
        logger.warning('Bivariate analysis not implemented yet')
        return frame, None

    def correlation(self, frame: pd.DataFrame) -> tuple:
        """Compute correlation between features and optionally drop highly correlated ones"""
        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.target_label, axis=1)
        corr_matrix = x_frame.corr(method=self.corr_method).round(2)

        if self.corr_drop_features:
            # calculate feature importances
            estimator = RandomForestClassifier(random_state=self.seed)
            estimator.fit(x_frame, y_frame)
            scoring = self.config.selection.scoring[self.learn_task]
            perm_importances = permutation_importance(estimator, x_frame, y_frame, scoring=scoring)
            importances = perm_importances.importances_mean
            importances = pd.Series(importances, index=x_frame.columns)
            importances = importances.sort_values(ascending=False)

            # sort corr_matrix w.r.t. feature importance
            corr_matrix = corr_matrix.reindex(index=importances.index, columns=importances.index)
            abs_corr = corr_matrix.abs()
            upper_tri = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))

            cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.corr_thresh)]
            x_frame = x_frame.drop(cols_to_drop, axis=1)
            logger.info(
                f'Removed {len(cols_to_drop)} redundant features with correlation above {self.corr_thresh}, '
                f'number of remaining features: {len(x_frame.columns)}'
            )
            corr_matrix = x_frame.corr(method=self.corr_method).round(2)

        # plot correlation heatmap
        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(corr_matrix, annot=True, xticklabels=True, yticklabels=True, cmap='viridis')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.job_dir, 'corr_plot.pdf'))
        plt.close(fig)
        frame = pd.concat((x_frame, frame[self.target_label]), axis=1)
        return frame, None

    def drop_outliers(self, frame: pd.DataFrame) -> tuple:
        """Detect outliers in the data, optionally removing or further investigating them"""
        whiskers = 1.5
        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.target_label, axis=1)
        # Calculate quartiles, interquartile range and limits
        q1, q3 = np.percentile(x_frame, [25, 75], axis=0)
        iqr = q3 - q1
        lower_limit = q1 - whiskers * iqr
        upper_limit = q3 + whiskers * iqr
        # logger.debug(f'\nlower limit: {lower_limit}\nupper limit: {upper_limit}')
        x_frame = x_frame.mask(x_frame.le(lower_limit) | x_frame.ge(upper_limit))
        x_frame.to_excel(os.path.join(self.job_dir, 'outliers_removed.xlsx'), index=True)
        dropped_outlier_frame = pd.concat((x_frame, y_frame), axis=1)
        return dropped_outlier_frame, None

    def detect_outliers(self, frame: pd.DataFrame) -> tuple:
        """Detect outliers in the data, optionally removing or further investigating them"""
        whiskers = 1.5
        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.metadata, axis=1, errors='ignore')

        # Calculate quartiles, interquartile range and limits
        q1, q3 = np.percentile(x_frame, [25, 75], axis=0)
        iqr = q3 - q1
        lower_limit = q1 - whiskers * iqr
        upper_limit = q3 + whiskers * iqr
        # logger.debug(f'\nlower limit: {lower_limit}\nupper limit: {upper_limit}')

        high_data = x_frame.copy(deep=True)
        # Highlight outliers in table
        high_data.style.apply(
            lambda _: self.highlight(frame=high_data, lower_limit=lower_limit, upper_limit=upper_limit), axis=None
        ).to_excel(os.path.join(self.job_dir, 'investigate_outliers.xlsx'), index=True)
        return frame, None

    def feature_wiz(self, frame: pd.DataFrame) -> tuple:
        """Use feature_wiz to select features"""
        from featurewiz import FeatureWiz

        y_train = frame[self.target_label]
        x_train = frame.drop(self.target_label, axis=1)

        features = FeatureWiz(
            corr_limit=self.corr_thresh,
            feature_engg='',
            category_encoders='',
            dask_xgboost_flag=False,
            nrows=None,
            verbose=2,
        )
        selected_features = features.fit_transform(x_train, y_train)

        return selected_features.join(y_train), selected_features.columns.tolist()

    def variance_threshold(self, frame: pd.DataFrame) -> tuple:
        """Remove features with variance below threshold"""
        y_frame = frame[self.target_label]

        selector = VarianceThreshold(threshold=self.corr_thresh * (1 - self.corr_thresh))
        selector.fit(frame)
        frame = frame.loc[:, selector.get_support()]
        logger.info(
            f'Removed {len(selector.get_support()) - len(frame.columns)} '
            f'features with same value in more than {int(self.corr_thresh*100)}% of subjects, '
            f'number of remaining features: {len(frame.columns)}'
        )

        if self.target_label not in frame.columns:  # ensure label col is kept
            logger.warning(f'Target label {self.target_label} has variance below threshold {self.corr_thresh}.')
            frame = pd.concat((frame, y_frame), axis=1)
        return frame, None
