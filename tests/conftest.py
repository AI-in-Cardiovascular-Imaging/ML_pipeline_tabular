from pytest import fixture

from cardio_parsers.stations.normalisers import Normaliser


@fixture(scope='function')
def normaliser():
    """Returns an unique normaliser for each function"""
    return Normaliser()
