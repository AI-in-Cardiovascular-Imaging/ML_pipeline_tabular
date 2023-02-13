import os
from copy import deepcopy
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from loguru import logger


def merge_metadata(data, mdata_src, metadata) -> pd.DataFrame:
    """Merge metadata with data"""
    metadata = ['pat_id'] + metadata  # always want patient ID
    mdata = pd.read_excel(mdata_src)
    mdata = mdata[metadata]
    mdata = mdata[mdata['pat_id'].notna()]  # remove rows without pat_id
    mdata = mdata.rename(columns={'pat_id': 'subject'})
    mdata['subject'] = mdata['subject'].astype(int)
    data['subject'] = data['subject'].astype(int)
    data = data.merge(mdata, how='left', on='subject')  # merge the cvi42 data with available metadata
    return data


def save_tables(src, experiment_name, tables) -> None:
    """Save tables to excel file"""
    file_path = os.path.join(src, '5_merged', f'{experiment_name}.xlsx')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tables.to_excel(file_path, index=False)


def split_data(data: pd.DataFrame, metadata: list, hue: str, remove_mdata: bool = True):
    """Split data into data to analyse and hue data"""
    to_analyse = deepcopy(data)
    hue_df = to_analyse[hue]
    if remove_mdata:
        to_analyse = to_analyse.drop(metadata, axis=1, errors='ignore')
    suffix = 'no_mdata' if remove_mdata else 'with_mdata'
    return to_analyse, hue_df, suffix


def normalize_data(data: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Normalise data"""
    tmp = data[target_label]  # keep label col as is
    # data = (data - data.mean()) / data.std()
    data = data / data.sum()
    data[target_label] = tmp
    return data


def variance_threshold(data: pd.DataFrame, label: str, thresh: float) -> pd.DataFrame:
    """Remove features with variance below threshold"""
    tmp = data[label]  # save label col
    logger.debug(len(data.columns))
    selector = VarianceThreshold(threshold=thresh * (1 - thresh))
    selector.fit(data)
    data = data.loc[:, selector.get_support()]

    logger.debug(len(data.columns))

    if label not in data.columns:  # ensure label col is kept
        logger.warning(f'Target label {label} has variance below threshold {thresh}.')
        data = pd.concat((data, tmp), axis=1)

    return data
