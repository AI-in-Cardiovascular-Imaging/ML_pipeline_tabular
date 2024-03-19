import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pipeline_tabular.data_handler.data_handler import DataHandler


class DataExploration(DataHandler):
    """Performs some data exploration and sets the learning task"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.plot_format = config.meta.plot_format
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.out_dir = config.meta.output_dir
        self.corr_method = config.selection.corr_method
        self.variance_thresh = config.selection.variance_thresh

    def __call__(self) -> None:
        frame = self.get_frame()
        target_frame = frame[self.target_label]
        imp_frame = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(frame)
        frame = pd.DataFrame(imp_frame, index=frame.index, columns=frame.columns)
        binary_cols = frame.nunique()[frame.nunique() == 2].index
        frame = frame.drop(binary_cols, axis=1)
        unary_cols = frame.nunique()[frame.nunique() == 1].index
        frame = frame.drop(unary_cols, axis=1)
        norm_frame = StandardScaler().fit_transform(frame)
        self.frame = pd.concat(
            [pd.DataFrame(norm_frame, index=frame.index, columns=frame.columns), target_frame], axis=1
        )
        self.corr_to_target()
        self.plot_cluster_map()
        self.plot_corr_heatmap()
        self.plot_stats()

    def corr_to_target(self) -> None:
        y = self.frame[self.target_label]
        x = self.frame.drop(self.target_label, axis=1)
        corr_series = x.corrwith(y, axis=0, method=self.corr_method).round(2)
        corr_df = pd.DataFrame({'correlation_to_target': corr_series.values, 'feature': corr_series.index})
        corr_df = corr_df.sort_values(by='correlation_to_target')
        corr_df.to_csv(
            os.path.join(self.out_dir, f'feature_{self.corr_method}_corr_target_{self.target_label}.txt'),
            sep='\t',
            header=True,
            index=False,
        )

    def plot_cluster_map(self) -> None:
        frame = self.frame.drop(self.target_label, axis=1)
        cluster_map = sns.clustermap(frame, figsize=(20, 20), cmap='coolwarm', method='ward', metric='euclidean')
        cluster_map.savefig(
            os.path.join(self.out_dir, f'feature_{self.corr_method}_cluster_map.{self.plot_format}'), dpi=300
        )
        plt.clf()

    def plot_corr_heatmap(self) -> None:
        frame = self.frame.drop(self.target_label, axis=1)
        corr_matrix = frame.corr(method=self.corr_method)
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_plot = sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='coolwarm',
            annot=False,
            square=True,
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            cbar_kws={'shrink': 0.5},
        )
        corr_plot.figure.tight_layout()
        corr_plot = corr_plot.get_figure()
        corr_plot.savefig(
            os.path.join(self.out_dir, f'feature_{self.corr_method}_corr_heatmap.{self.plot_format}'), dpi=300
        )

    def plot_stats(self) -> None:
        """Plot target statistics"""
        if self.target_label not in self.frame.columns:
            raise ValueError(f'Target label {self.target_label} not in data')
        self.target_frame = self.frame[self.target_label]
        if self.learn_task == 'binary_classification':
            perc = int((self.target_frame.sum() / len(self.target_frame.index)).round(2) * 100)
            logger.info(
                f'\nSummary statistics for binary target variable {self.target_label}:\n'
                f'Positive cases -> {perc}% or {int(self.target_frame.sum())}/{len(self.target_frame.index)} samples.'
            )

        elif self.learn_task == 'multi_classification':
            raise NotImplementedError('Multi-classification is not implemented yet')

        elif self.learn_task == 'regression':
            logger.info(
                f'\nSummary statistics for continuous target variable {self.target_label}:\n'
                f'{self.target_frame.describe(percentiles=[]).round(2)}'
            )

        else:
            raise ValueError(f'Unknown learn task: {self.learn_task}')
