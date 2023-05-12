import os
import mrmr

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
        self.workers = None
        self.target_label = None
        self.corr_method = None
        self.corr_thresh = None
        self.corr_ranking = None
        self.variance_thresh = None
        self.scoring = None
        self.learn_task = None
        self.n_top_features = None
        np.random.seed(self.seed)

    def variance_threshold(self, frame: pd.DataFrame) -> tuple:
        """Remove features with variance below threshold"""
        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.target_label, axis=1)
        selector = VarianceThreshold(threshold=self.variance_thresh * (1 - self.variance_thresh))
        selector.fit(x_frame)
        x_frame = x_frame.loc[:, selector.get_support()]
        logger.info(
            f'Removed {len(selector.get_support()) - len(x_frame.columns)} '
            f'features with same value in more than {int(self.variance_thresh*100)}% of subjects, '
            f'number of remaining features: {len(x_frame.columns)}'
        )
        new_frame = pd.concat([x_frame, y_frame], axis=1)
        features = list(x_frame.columns)
        return new_frame, features

    def correlation(self, frame: pd.DataFrame) -> tuple:
        """Compute correlation between features and optionally drop highly correlated ones"""
        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.target_label, axis=1)
        corr_matrix = x_frame.corr(method=self.corr_method).round(2)

        # calculate feature importance
        if self.corr_ranking == 'forest':
            estimator = RandomForestClassifier(random_state=self.seed, n_jobs=self.workers)
            estimator.fit(x_frame, y_frame)
            scoring = self.config.selection.scoring[self.learn_task]
            perm_importances = permutation_importance(
                estimator, x_frame, y_frame, n_repeats=5, scoring=scoring, random_state=self.seed, n_jobs=self.workers
            )
            importances = perm_importances.importances_mean
            importances = pd.Series(importances, index=x_frame.columns)
        elif self.corr_ranking == 'corr':
            importances = x_frame.corrwith(y_frame, axis=0, method=self.corr_method).round(2)
            importances = importances.abs()
        else:
            logger.error(f'Selected corr_ranking method {self.corr_ranking} has not been implemented.')
            raise NotImplementedError
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
        abs_corr = abs_corr.drop(cols_to_drop, axis=0)
        abs_corr = abs_corr.drop(cols_to_drop, axis=1)

        # plot correlation heatmap
        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(abs_corr, annot=False, xticklabels=True, yticklabels=True, cmap='viridis')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.job_dir, 'corr_plot.pdf'))
        plt.close(fig)

        new_frame = pd.concat([x_frame, y_frame], axis=1)
        features = list(x_frame.columns)
        return new_frame, features

    def mrmr(self, frame: pd.DataFrame) -> tuple:
        """Maximum relevance minimum redundancy to select features"""
        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.target_label, axis=1)
        nunique = self.frame.nunique()
        categorical = nunique[nunique <= 10].index
        if self.learn_task == 'classification':
            features = mrmr.mrmr_classif(
                x_frame,
                y_frame,
                K=self.n_top_features,
                cat_features=categorical,
                n_jobs=self.workers,
                show_progress=False,
            )
        else:
            features = mrmr.mrmr_regression(
                x_frame,
                y_frame,
                K=self.n_top_features,
                cat_features=categorical,
                n_jobs=self.workers,
                show_progress=False,
            )

        x_frame = x_frame[features]
        new_frame = pd.concat([x_frame, y_frame], axis=1)
        return new_frame, features

    def feature_wiz(self, frame: pd.DataFrame) -> tuple:
        """Use feature_wiz to select features"""
        from featurewiz import FeatureWiz

        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.target_label, axis=1)

        features = FeatureWiz(
            corr_limit=self.corr_thresh,
            feature_engg='',
            category_encoders='',
            dask_xgboost_flag=False,
            nrows=None,
            verbose=2,
        )
        selected_features = features.fit_transform(x_frame, y_frame)
        new_frame = pd.concat([selected_features, y_frame], axis=1)
        features = selected_features.columns.tolist()
        return new_frame, features

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

    def bivariate_analysis(self, frame: pd.DataFrame) -> tuple:
        """Perform bivariate analysis"""
        logger.warning('Bivariate analysis not implemented yet')
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

        high_data = pd.concat([x_frame, y_frame], axis=1)
        # Highlight outliers in table
        high_data.style.apply(
            lambda _: self.highlight(frame=high_data, lower_limit=lower_limit, upper_limit=upper_limit), axis=None
        ).to_excel(os.path.join(self.job_dir, 'investigate_outliers.xlsx'), index=True)
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
        return frame, None
