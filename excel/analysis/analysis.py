""" Analysis module for all kinds of experiments
"""

import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import pandas as pd

from excel.analysis.utils.merge_data import MergeData
from excel.analysis.utils.update_metadata import UpdateMetadata
from excel.analysis.utils.exploration import ExploreData

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


@hydra.main(version_base=None, config_path='../../config', config_name='config')
def analysis(config: DictConfig) -> None:
    """Analysis pipeline

    Args:
        config (DictConfig): config element containing all config parameters
            check the config files for info on the individual parameters

    Returns:
        None
    """
    # Parse some config parameters
    src_dir = config.dataset.out_dir
    mdata_src = config.dataset.mdata_src
    dims = config.dataset.dims
    strict = config.dataset.strict

    segments = config.analysis.segments
    axes = config.analysis.axes
    orientations = config.analysis.orientations
    metrics = config.analysis.metrics
    peak_values = config.analysis.peak_values
    impute = config.analysis.impute
    metadata = config.analysis.metadata
    experiment = config.analysis.experiment
    overwrite = config.analysis.overwrite
    update_metadata = config.analysis.update_metadata
    exploration = config.analysis.exploration
    remove_outliers = config.analysis.remove_outliers
    investigate_outliers = config.analysis.investigate_outliers
    whis = config.analysis.whis
    seed = config.analysis.seed
    corr_thresh = config.analysis.corr_thresh
    drop_features = config.analysis.drop_features
    feature_reduction = config.analysis.feature_reduction

    # TODO: train and test paths/sets
    experiment = experiment + '_imputed' if impute else experiment
    merged_path = os.path.join(src_dir, '5_merged', f'{experiment}.xlsx')

    # Data merging
    if os.path.isfile(merged_path) and not overwrite:
        logger.info('Merged data available, skipping merge step...')
    else:
        logger.info('Merging data according to config parameters...')
        merger = MergeData(
            src=src_dir,
            mdata_src=mdata_src,
            dims=dims,
            strict=strict,
            segments=segments,
            axes=axes,
            orientations=orientations,
            metrics=metrics,
            peak_values=peak_values,
            metadata=metadata,
            experiment=experiment,
            impute=impute,
            seed=seed,
        )
        merger()
        logger.info('Data merging finished.')

    # Read in merged data
    data = pd.read_excel(merged_path)

    # Update metadata if desired (only makes sense if overwrite=False)
    if not overwrite and update_metadata:
        logger.info('Updating metadata as requested...')
        updater = UpdateMetadata(src=src_dir, data=data, mdata_src=mdata_src, metadata=metadata, experiment=experiment)
        data = updater()
        logger.info('Metadata update finished.')

    # Use subject ID as index column
    data = data.set_index('subject')

    # Data exploration
    if exploration:
        expl_dir = os.path.join(src_dir, '6_exploration', experiment)
        os.makedirs(expl_dir, exist_ok=True)

        explorer = ExploreData(
            data=data,
            experiment=experiment,
            exploration=exploration,
            out_dir=expl_dir,
            remove_outliers=remove_outliers,
            investigate_outliers=investigate_outliers,
            metadata=metadata,
            whis=whis,
            seed=seed,
            corr_thresh=corr_thresh,
            drop_features=drop_features,
            feature_reduction=feature_reduction,
        )
        explorer()


if __name__ == '__main__':
    analysis()
