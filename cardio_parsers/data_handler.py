import copy
import json
from collections import defaultdict

import pandas as pd
from loguru import logger


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded"""

    def __init__(self, *args, **kwargs):
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


class DataHandler:

    shared_state = {
        '__data_store': NestedDefaultDict(),
        '__original_data': None,
        '__ephemeral_data': None,
        '__feature_store': NestedDefaultDict(),
        '__state_name': None,
    }

    def __init__(self) -> None:
        self.__dict__ = self.shared_state

    def show(self) -> None:
        """Prints the data store"""
        print(json.dumps(self.__data_store, indent=4, sort_keys=True))

    def set_state_name(self, state_name: str) -> None:
        """Sets the state name"""
        if isinstance(state_name, tuple):
            self.__state_name = '_'.join(state_name)
            self.__data_store[self.__state_name] = NestedDefaultDict()
            logger.info(f'State name set to -> {self.__state_name}')
        else:
            raise ValueError(f'Invalid state name type, found -> {type(state_name)}, allowed -> tuple')

    def set_state_values(self, key: str, frame: pd.DataFrame) -> None:
        """Sets the state value"""
        self.__ephemeral_data = copy.deepcopy(frame)
        self.__data_store[self.__state_name] = {key: frame}
        logger.info(f'State value set to -> {type(key)}')

    def get_ephemeral_data(self) -> pd.DataFrame:
        """Returns the ephemeral data"""
        logger.info(f'Returning ephemeral data -> {type(self.__ephemeral_data)}')
        return self.__ephemeral_data

    def set_original_data(self, frame: pd.DataFrame) -> None:
        """Sets the original data"""
        self.__original_data = frame
        self.__ephemeral_data = copy.deepcopy(frame)
        logger.info(f'Original data set to -> {type(self.__original_data)}')
        logger.info(f'Ephemeral data set to -> {type(self.__ephemeral_data)}')
