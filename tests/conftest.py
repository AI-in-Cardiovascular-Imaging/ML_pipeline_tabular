from cardio_parsers.stations.normalisers import Normaliser
from pytest import fixture


@fixture(scope='function')
def normaliser():
    """Returns an unique normaliser for each function"""
    return Normaliser()
