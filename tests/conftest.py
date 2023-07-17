from omegaconf import OmegaConf
from pytest import fixture

from feature_corr.crates.data_split import DataSplit
from feature_corr.crates.normalisers import Normalisers


@fixture(scope='function')
def normaliser():
    """Returns an unique normaliser for each function"""
    return Normalisers()


@fixture(scope='function')
def data_split():
    """Returns as unique function"""
    config = OmegaConf.create(
        {
            'meta': {
                'seed': 42,
                'state_name': 'test',
                'learn_task': 'binary_classification',
                'target_label': 'A',
            },
            'data_split': {
                'over_sample_selection': False,
                'over_sample_verification': False,
                'selection_frac': 0.4,
                'test_frac': 0.4,
                'over_sample_method': {
                    "binary_classification": "RandomOverSampler",
                    "multi_classification": "RandomOverSampler",
                    "regression": "RandomOverSampler",
                },
            },
        }
    )
    return DataSplit(config)
