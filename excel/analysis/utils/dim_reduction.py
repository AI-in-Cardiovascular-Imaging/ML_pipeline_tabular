"""Dimensionality reduction module
"""

import os
from copy import deepcopy

from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from excel.analysis.utils.helpers import split_data


class DimensionReductions:

    def __init__(self) -> None:
        self.job_dir = None
        self.metadata = None
        self.seed = None
        self.target_label = None

    def pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform Principal Component Analysis (PCA)

        Args:
            data (pd.DataFrame): DataFrame containing all relevant features and metadata (opt.)
            out_dir (str): directory to store plots
            metadata (list): list of metadata column names
            seed (int): random seed
        """
        to_analyse = deepcopy(data)
        # OPT: could be removed (at least for impute=True)
        to_analyse = to_analyse.dropna(how='any')  # drop rows containing any NaN values
        to_analyse, hue_df, suffix = split_data(to_analyse, self.metadata, self.target_label, remove_mdata=False)  # split data and metadata
        to_analyse = to_analyse.drop(self.target_label, axis=1)
        pca = PCA(n_components=4, random_state=self.seed)  # perform PCA
        analysis = pca.fit_transform(to_analyse)
        analysis = pd.DataFrame(analysis, index=to_analyse.index, columns=['pc_1', 'pc_2', 'pc_3', 'pc_4'])
        analysis = pd.concat((analysis, hue_df), axis=1)
        explained_var = pca.explained_variance_ratio_
        logger.info(f'Variance explained: {explained_var} for {len(analysis.index)} subjects ({suffix}).')
        # logger.info(f'\n{abs(pca.components_)}')

        sns.lmplot(
            data=analysis,
            x='pc_1',
            y='pc_2',
            hue=self.target_label,
            fit_reg=False,
            legend=True,
            scatter_kws={'s': 5, 'marker': '*', 'alpha': 0.5},
            x_jitter=0.2,
            y_jitter=0.2,
        )
        # plt.tight_layout()
        plt.xlabel(f'First PC (explains {int(round(explained_var[0], 2) * 100)}% of variance)')
        plt.ylabel(f'Second PC (explains {int(round(explained_var[1], 2) * 100)}% of variance)')
        plt.title('Principal Component Analysis')
        plt.savefig(os.path.join(self.job_dir, f'PCA_{suffix}.pdf'), bbox_inches='tight')
        plt.clf()
        return data

    def tsne(self,data: pd.DataFrame) -> pd.DataFrame:
        """Perform t-SNE dimensionality reduction and visualisation

        Args:
            data (pd.DataFrame): DataFrame containing all relevant features and metadata (opt.)
            out_dir (str): directory to store plots
            metadata (list): list of metadata column names
            seed (int): random seed
        """
        to_analyse = data.copy(deep=True)
        to_analyse = to_analyse.dropna(how='any')  # drop rows containing any NaN values
        to_analyse, hue_df, suffix = split_data(to_analyse, self.metadata, self.target_label, remove_mdata=False)
        to_analyse = to_analyse.drop(self.target_label, axis=1)
        perplexities = [5, 15, 30, 50]  # perplexities to test

        for perp in perplexities:  # perform t-SNE for different perplexities
            tsne = TSNE(n_components=2, perplexity=perp, random_state=self.seed)
            analysis = tsne.fit_transform(to_analyse)
            analysis = pd.DataFrame(analysis, index=to_analyse.index, columns=['tsne_1', 'tsne_2'])
            analysis = pd.concat((analysis, hue_df), axis=1)

            sns.lmplot(data=analysis, x='tsne_1', y='tsne_2', hue=self.target_label, fit_reg=False, legend=True, scatter_kws={'s': 20})
            plt.title(f't-SNE for perplexity {perp}')
            plt.savefig(os.path.join(self.job_dir, f'TSNE_{suffix}_perp_{perp}.pdf'), bbox_inches='tight')
            plt.clf()

        logger.info(f'{len(analysis.index)} subjects ({suffix}).')
        return data

    def umap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform UMAP dimensionality reduction and visualisation

        Args:
            data (pd.DataFrame): DataFrame containing all relevant features and metadata (opt.)
            out_dir (str): directory to store plots
            metadata (list): list of metadata column names
            seed (int): random seed
        """
        to_analyse = data.copy(deep=True)
        to_analyse = to_analyse.dropna(how='any')  # drop rows containing any NaN values

        to_analyse, hue_df, suffix = split_data(to_analyse, self.metadata, self.target_label, remove_mdata=False)
        to_analyse = to_analyse.drop(self.target_label, axis=1)

        reducer = UMAP(n_components=2, random_state=self.seed)
        analysis = reducer.fit_transform(to_analyse)
        analysis = pd.DataFrame(analysis, index=to_analyse.index, columns=['umap_1', 'umap_2'])
        analysis = pd.concat((analysis, hue_df), axis=1)
        sns.lmplot(
            data=analysis,
            x='umap_1',
            y='umap_2',
            hue=self.target_label,
            fit_reg=False,
            legend=True,
            scatter_kws={'s': 20},
            x_jitter=0.2,
            y_jitter=0.2,
        )
        plt.title(f'Umap analysis')
        plt.savefig(os.path.join(self.job_dir, f'Umap_{suffix}.pdf'), bbox_inches='tight')
        plt.clf()
        logger.info(f'{len(analysis.index)} subjects ({suffix}).')
        return data
