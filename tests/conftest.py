from pytest import fixture

from feature_corr.crates.normalisers import Normalisers


@fixture(scope='function')
def normaliser():
    """Returns an unique normaliser for each function"""
    return Normalisers()
