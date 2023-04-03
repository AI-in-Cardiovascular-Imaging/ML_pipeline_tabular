import os

import pandas as pd
import plotly.express as px
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def plot_bubble(func):
    """Creates 2D and 3D scatter plots of the frame"""

    def wrapper(self, *args):
        frame = args[0]
        show_plots = self.config.meta.pipeline_plots
        y_train = frame[self.target_label]
        x_train = frame.drop(self.target_label, axis=1)

        proj_2d, proj_3d, name = func(self, x_train)  # call the wrapped function
        if show_plots:
            if proj_2d is not None:
                fig_2d = px.scatter(
                    proj_2d,
                    x=0,
                    y=1,
                    color=y_train,
                    labels={'color': self.target_label},
                    title=f'{name} 2D',
                    color_continuous_scale='viridis',
                )
                fig_2d.write_image(os.path.join(self.job_dir, f'{name}_2d.svg'))
                fig_2d.write_html(os.path.join(self.job_dir, f'{name}_2d.html'))
            else:
                logger.warning(f'Cannot plot {name} 2D, needs at least 2 features to run')

            if proj_3d is not None:
                fig_3d = px.scatter_3d(
                    proj_3d,
                    x=0,
                    y=1,
                    z=2,
                    color=y_train,
                    labels={'color': self.target_label},
                    title=f'{name} 3D',
                    color_continuous_scale='viridis',
                )
                fig_3d.update_traces(marker_size=5)
                fig_3d.write_html(os.path.join(self.job_dir, f'{name}_3d.html'))
            else:
                logger.warning(f'Cannot plot {name} 3D, needs at least 3 features to run')

        if proj_2d is not None:
            proj_2d = pd.DataFrame(proj_2d, index=frame.index)
            frame = pd.concat([proj_2d, y_train], axis=1)
        if proj_3d is not None:
            proj_3d = pd.DataFrame(proj_3d, index=frame.index)
            frame = pd.concat([proj_3d, y_train], axis=1)

        return frame, None

    return wrapper


class DimensionProjections:
    """Dimensionality reduction and visualisation"""

    def __init__(self) -> None:
        self.job_dir = None
        self.metadata = None
        self.seed = None
        self.workers = None
        self.target_label = None

    @plot_bubble
    def pca(self, frame: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, str):
        """Perform Principal Component Analysis (PCA)"""
        proj_2d, proj_3d = None, None
        if len(frame.columns) >= 2:
            pca_2d = PCA(n_components=2, random_state=self.seed)
            proj_2d = pca_2d.fit_transform(frame)
        if len(frame.columns) >= 3:
            pca_3d = PCA(n_components=3, random_state=self.seed)
            proj_3d = pca_3d.fit_transform(frame)
        return proj_2d, proj_3d, 'PCA'

    @plot_bubble
    def tsne(self, frame: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, str):
        """Perform t-SNE dimensionality reduction and visualisation"""
        proj_2d, proj_3d = None, None
        if len(frame.columns) >= 2:
            tsne_2d = TSNE(n_components=2, random_state=self.seed, n_jobs=self.workers)
            proj_2d = tsne_2d.fit_transform(frame)
        if len(frame.columns) >= 3:
            tsne_3d = TSNE(n_components=3, random_state=self.seed, n_jobs=self.workers)
            proj_3d = tsne_3d.fit_transform(frame)
        return proj_2d, proj_3d, 't-SNE'

    @plot_bubble
    def umap(self, frame: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, str):
        """Perform UMAP dimensionality reduction and visualisation"""
        proj_2d, proj_3d = None, None
        if len(frame.columns) >= 2:
            umap_2d = UMAP(n_components=2, random_state=self.seed)
            proj_2d = umap_2d.fit_transform(frame)
        if len(frame.columns) >= 3:
            umap_3d = UMAP(n_components=3, random_state=self.seed)
            proj_3d = umap_3d.fit_transform(frame)
        return proj_2d, proj_3d, 'UMAP'
