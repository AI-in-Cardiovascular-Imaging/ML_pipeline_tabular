from collections import defaultdict

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
        '_frame_store': NestedDefaultDict(),
        '_feature_store': NestedDefaultDict(),
        '_original_frame': None,
        '_ephemeral_frame': None,
    }

    def __init__(self) -> None:
        self._frame_store = NestedDefaultDict()
        self._feature_store = NestedDefaultDict()
        self._original_frame = None
        self._ephemeral_frame = None
        self.__dict__ = self.shared_state  # borg design pattern

    def add_state_name(self, state_name: str) -> None:
        """Sets the state name"""
        self._frame_store[state_name] = NestedDefaultDict()
        self._feature_store[state_name] = NestedDefaultDict()
        logger.trace(f'State name set -> {state_name}')

    def set_frame(self, data_name: str, frame: pd.DataFrame) -> None:
        """Sets the data"""
        if 'original' in data_name:
            self._original_frame = frame
            logger.trace(f'Original data set -> {type(frame)}')
        elif 'ephemeral' in data_name:
            self._ephemeral_frame = frame
            logger.trace(f'Ephemeral data set -> {type(frame)}')
        else:
            raise ValueError(f'Invalid name -> {data_name}, allowed -> original, ephemeral')

    def get_frame(self, data_name: str) -> pd.DataFrame:
        """Returns the data"""
        if 'original' in data_name:
            logger.trace(f'Returning original data -> {type(self._original_frame)}')
            return self._original_frame
        if 'ephemeral' in data_name:
            logger.trace(f'Returning ephemeral data -> {type(self._ephemeral_frame)}')
            return self._ephemeral_frame
        raise ValueError(f'Invalid name -> {data_name}, allowed -> original, ephemeral')

    def set_store(self, data_name: str, state_name: str, step_name: str, frame: pd.DataFrame) -> None:
        """Sets the store data"""
        if 'frame' in data_name:
            self._frame_store[state_name][step_name] = frame
            logger.trace(f'Store data set -> {type(frame)}')
        elif 'feature' in data_name:
            self._feature_store[state_name][step_name] = frame
            logger.trace(f'Feature data set -> {type(frame)}')
        else:
            raise ValueError(f'Invalid data name to set store data -> {data_name}, allowed -> data, feature')

    def get_store(self, data_name: str, state_name: str, step_name: str) -> pd.DataFrame:
        """Returns the store value"""
        if 'frame' in data_name:
            logger.trace(f'Returning data -> {type(self._frame_store[state_name][step_name])}')
            return self._frame_store[state_name][step_name]
        if 'feature' in data_name:
            logger.trace(f'Returning feature -> {type(self._feature_store[state_name][step_name])}')
            return self._feature_store[state_name][step_name]
        raise ValueError(f'Invalid data name to get store data -> {data_name}, allowed -> data, feature')

    def sync_ephemeral_data_to_data_store(self, state_name: str, step_name: str) -> None:
        """Syncs the ephemeral data with the data store"""
        self._frame_store[state_name][step_name] = self._ephemeral_frame
        logger.trace(f'Ephemeral data synced -> {type(self._ephemeral_frame)} to data store')

    def remove_state_data_store(self, state_name: str) -> None:
        """Removes the state, prevent memory overflows"""
        if state_name in self._frame_store:
            del self._frame_store[state_name]
            logger.trace(f'State removed -> {state_name}')
        else:
            raise ValueError(f'Invalid state name -> {state_name}')
