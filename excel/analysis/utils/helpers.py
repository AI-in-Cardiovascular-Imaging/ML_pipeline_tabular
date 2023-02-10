import os

from loguru import logger
import pandas as pd


def merge_metadata(data, mdata_src, metadata) -> pd.DataFrame:
    # Always want patient ID
    metadata = ['pat_id'] + metadata
    mdata = pd.read_excel(mdata_src)
    mdata = mdata[metadata]
    mdata = mdata[mdata['pat_id'].notna()]  # remove rows without pat_id
    mdata = mdata.rename(columns={'pat_id': 'subject'})
    mdata['subject'] = mdata['subject'].astype(int)
    data['subject'] = data['subject'].astype(int)

    # Merge the cvi42 data with available metadata
    data = data.merge(mdata, how='left', on='subject')

    return data


def save_tables(src, experiment, tables) -> None:
    file_path = os.path.join(src, '5_merged', f'{experiment}.xlsx')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tables.to_excel(file_path, index=False)


def split_data(data: pd.DataFrame, metadata: list, hue: str, remove_mdata: bool = True):
    to_analyse = data.copy(deep=True)
    hue_df = to_analyse[hue]
    
    if remove_mdata:
        to_analyse = to_analyse.drop(metadata, axis=1)

    suffix = 'no_mdata' if remove_mdata else 'with_mdata'

    return to_analyse, hue_df, suffix

def normalise_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
    tmp = data[label] # keep label col as is
    # data = (data - data.mean()) / data.std()
    data = data / data.sum()
    data[label] = tmp

    return data
