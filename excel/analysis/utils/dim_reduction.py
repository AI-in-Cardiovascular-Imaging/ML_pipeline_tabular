"""Dimensionality reduction module
"""

import os

from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from excel.analysis.utils.helpers import split_data


def pca(data: pd.DataFrame, out_dir: str, metadata: list, hue: str, seed: int):
    """Perform Principal Component Analysis (PCA)

    Args:
        data (pd.DataFrame): DataFrame containing all relevant features and metadata (opt.)
        out_dir (str): directory to store plots
        metadata (list): list of metadata column names
        seed (int): random seed
    """
    for remove_mdata in [True, False]:
        try:
            to_analyse = data.copy(deep=True)
            # OPT: could be removed (at least for impute=True)
            to_analyse = to_analyse.dropna(how='any')  # drop rows containing any NaN values
            # Split data and metadata
            to_analyse, hue_df, suffix = split_data(
                to_analyse, metadata, hue, remove_mdata=remove_mdata, normalise=True
            )

            # Perform PCA
            pca = PCA(n_components=4)
            analysis = pca.fit_transform(to_analyse)
            analysis = pd.DataFrame(analysis, index=to_analyse.index, columns=['pc_1', 'pc_2', 'pc_3', 'pc_4'])
            analysis = pd.concat((analysis, hue_df), axis=1)
            explained_var = pca.explained_variance_ratio_
            logger.info(f'Variance explained: {explained_var} for {len(analysis.index)} subjects ({suffix}).')
            # logger.info(f'\n{abs(pca.components_)}')

            # Plot the transformed dataset
            sns.lmplot(data=analysis, x='pc_1', y='pc_2', hue='mace', fit_reg=False, legend=True, scatter_kws={'s': 20})
            # plt.tight_layout()
            plt.xlabel(f'First PC (explains {int(round(explained_var[0], 2) * 100)}% of variance)')
            plt.ylabel(f'Second PC (explains {int(round(explained_var[1], 2) * 100)}% of variance)')
            plt.title('Principal Component Analysis')
            plt.savefig(os.path.join(out_dir, f'PCA_{suffix}.pdf'), bbox_inches='tight')
            plt.clf()
        except ValueError:
            pass


def tsne(data: pd.DataFrame, out_dir: str, metadata: list, hue: str, seed: int):
    """Perform t-SNE dimensionality reduction and visualisation

    Args:
        data (pd.DataFrame): DataFrame containing all relevant features and metadata (opt.)
        out_dir (str): directory to store plots
        metadata (list): list of metadata column names
        seed (int): random seed
    """
    for remove_mdata in [True, False]:
        try:
            to_analyse = data.copy(deep=True)
            to_analyse = to_analyse.dropna(how='any')  # drop rows containing any NaN values

            to_analyse, hue_df, suffix = split_data(
                to_analyse, metadata, hue, remove_mdata=remove_mdata, normalise=True
            )

            # Perform t-SNE for different perplexities
            perplexities = [5, 15, 30, 50]

            for perp in perplexities:
                # Calculate t-SNE for given perplexity
                tsne = TSNE(n_components=2, perplexity=perp, random_state=seed)
                analysis = tsne.fit_transform(to_analyse)
                analysis = pd.DataFrame(analysis, index=to_analyse.index, columns=['tsne_1', 'tsne_2'])
                analysis = pd.concat((analysis, hue_df), axis=1)

                # Plot the transformed dataset
                sns.lmplot(
                    data=analysis, x='tsne_1', y='tsne_2', hue='mace', fit_reg=False, legend=True, scatter_kws={'s': 20}
                )
                plt.title(f't-SNE for perplexity {perp}')
                plt.savefig(os.path.join(out_dir, f'TSNE_{suffix}_perp_{perp}.pdf'), bbox_inches='tight')
                plt.clf()

            logger.info(f'{len(analysis.index)} subjects ({suffix}).')
        except ValueError:
            pass
