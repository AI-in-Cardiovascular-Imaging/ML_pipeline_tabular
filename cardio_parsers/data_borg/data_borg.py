from collections import defaultdict
import copy

import pandas as pd
from loguru import logger


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded"""

    def __init__(self, *args, **kwargs):
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


class DataBorg:
    """Borg pattern, which is used to share data between classes"""

    shared_state = {
        '_data_store': NestedDefaultDict(),
        '_feature_store': NestedDefaultDict(),
        '_original_data': None,
    }

    def __init__(self) -> None:
        self._data_store = NestedDefaultDict()
        self._feature_store = NestedDefaultDict()
        self._original_data = None
        self.__dict__ = self.shared_state  # borg design pattern

    def add_state_name(self, state_name: str) -> None:
        """Sets the state name"""
        self._data_store[state_name] = NestedDefaultDict()
        logger.info(f'State name set -> {state_name}')

    def set_ephemeral_data(self, state_name: str, frame: pd.DataFrame) -> None:
        """Sets the ephemeral data"""
        if state_name in self._data_store:
            self._data_store[state_name]['ephemeral'] = frame
            logger.trace(f'Ephemeral data set -> {type(frame)}')
        else:
            raise ValueError(f'Invalid state name -> {state_name}')

    def set_store_data(self, state_name: str, step_name: str, frame: pd.DataFrame) -> None:
        """Sets the state value"""
        if state_name in self._data_store:
            self._data_store[state_name][step_name] = frame
            logger.trace(f'Store data set -> {type(frame)}')
        else:
            raise ValueError(f'Invalid state name -> {state_name}')

    def set_feature_store(self, state_name: str, frame: pd.DataFrame) -> None:
        """Sets the feature store"""
        if state_name in self._feature_store:
            self._feature_store[state_name] = frame
            logger.trace(f'Feature store set -> {type(frame)}')
        else:
            raise ValueError(f'Invalid state name -> {state_name}')

    def get_ephemeral_data(self, state_name: str) -> pd.DataFrame:
        """Returns the ephemeral data"""
        if state_name in self._data_store:
            logger.trace(f'Returning ephemeral data -> {type(self._data_store[state_name]["ephemeral"])}')
            return self._data_store[state_name]['ephemeral']
        else:
            raise ValueError(f'Invalid state name -> {state_name}')

    def get_store_data(self, state_name: str, step_name: str) -> pd.DataFrame:
        """Returns the state value"""
        if state_name in self._data_store:
            logger.trace(f'Returning store data -> {type(self._data_store[state_name][step_name])}')
            return self._data_store[state_name][step_name]
        else:
            raise ValueError(f'Invalid state name -> {state_name}')

    def get_feature_store(self, state_name: str) -> pd.DataFrame:
        """Returns the feature store"""
        if state_name in self._feature_store:
            logger.trace(f'Returning feature store -> {type(self._feature_store[state_name])}')
            return self._feature_store[state_name]
        else:
            raise ValueError(f'Invalid state name -> {state_name}')

    def set_original_data(self, frame: pd.DataFrame) -> None:
        """Sets the original data"""
        self._original_data = frame
        logger.trace(f'Original data set to -> {type(self._original_data)}')

    def get_original_data(self) -> pd.DataFrame:
        """Returns the original data"""
        logger.trace(f'Returning original data -> {type(self._original_data)}')
        return self._original_data

    def copy_original_to_ephemeral(self, state_name: str) -> None:
        """Copies the original data to ephemeral"""
        self._data_store[state_name]['ephemeral'] = copy.deepcopy(self._original_data)
        logger.trace(f'Original data copied to ephemeral -> {type(self._data_store[state_name]["ephemeral"])}')

    def remove_state_data(self, state_name: str) -> None:
        """Removes the state, prevent memory overflows"""
        if state_name in self._data_store:
            del self._data_store[state_name]
            logger.trace(f'State removed -> {state_name}')
        else:
            raise ValueError(f'Invalid state name -> {state_name}')
