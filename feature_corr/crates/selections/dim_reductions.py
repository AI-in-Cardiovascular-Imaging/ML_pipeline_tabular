import os

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def plot_bubble(func):
    """Creates 2D and 3D scatter plots of the frame"""

    def wrapper(self, *args):
        frame = args[0]
        y_train = frame[self.target_label]
        x_train = frame.drop(self.target_label, axis=1)

        proj_2d, proj_3d, name = func(self, x_train)  # call the wrapped function

        fig_2d = px.scatter(
            proj_2d,
            x=0,
            y=1,
            color=y_train,
            labels={'color': self.target_label},
            title=f'{name} 2D',
        )
        fig_3d = px.scatter_3d(
            proj_3d,
            x=0,
            y=1,
            z=2,
            color=y_train,
            labels={'color': self.target_label},
            title=f'{name} 3D',
        )
        fig_3d.update_traces(marker_size=5)

        fig_2d.write_image(os.path.join(self.job_dir, f'{name}_2d.svg'))
        fig_2d.write_html(os.path.join(self.job_dir, f'{name}_2d.html'))
        fig_3d.write_html(os.path.join(self.job_dir, f'{name}_3d.html'))

        return data

    return wrapper


class DimensionReductions:
    """Dimensionality reduction and visualisation"""

    def __init__(self) -> None:
        self.job_dir = None
        self.metadata = None
        self.seed = None
        self.target_label = None

    @plot_bubble
    def pca(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, str):
        """Perform Principal Component Analysis (PCA)"""
        pca_2d = PCA(n_components=2, random_state=self.seed)
        pca_3d = PCA(n_components=3, random_state=self.seed)
        proj_2d = pca_2d.fit_transform(data)
        proj_3d = pca_3d.fit_transform(data)
        return proj_2d, proj_3d, 'PCA'

    @plot_bubble
    def tsne(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, str):
        """Perform t-SNE dimensionality reduction and visualisation"""
        tsne_2d = TSNE(n_components=2, random_state=self.seed)
        tsne_3d = TSNE(n_components=3, random_state=self.seed)
        proj_2d = tsne_2d.fit_transform(data)
        proj_3d = tsne_3d.fit_transform(data)
        return proj_2d, proj_3d, 't-SNE'

    @plot_bubble
    def umap(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, str):
        """Perform UMAP dimensionality reduction and visualisation"""
        umap_2d = UMAP(n_components=2, random_state=self.seed)
        umap_3d = UMAP(n_components=3, random_state=self.seed)
        proj_2d = umap_2d.fit_transform(data)
        proj_3d = umap_3d.fit_transform(data)
        return proj_2d, proj_3d, 'UMAP'
