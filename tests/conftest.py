from pytest import fixture

from excel.analysis.utils.normalisers import Normaliser


@fixture(scope='function')
def normaliser():
    """Returns an unique normaliser for each function"""
    return Normaliser()
