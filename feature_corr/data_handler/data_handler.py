import os
import json

import pandas as pd
from collections import defaultdict
from loguru import logger


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded"""

    def __init__(self, *args, **kwargs):
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


class DataHandler:
    """Borg pattern, which is used to share frame between classes"""

    shared_state = {
        '_frame_store': NestedDefaultDict(),
        '_feature_store': NestedDefaultDict(),
        '_feature_score_store': NestedDefaultDict(),
        '_score_store': NestedDefaultDict(),
        '_original_frame': None,
        '_ephemeral_frame': None,
    }

    def __init__(self) -> None:
        self._frame_store = NestedDefaultDict()
        self._feature_store = NestedDefaultDict()
        self._feature_score_store = NestedDefaultDict()
        self._score_store = NestedDefaultDict()
        self._original_frame = None
        self._ephemeral_frame = None
        self.__dict__ = self.shared_state  # borg design pattern

    def add_state_name(self, state_name: str) -> None:
        """Sets the state name"""
        self._frame_store[state_name] = NestedDefaultDict()
        self._feature_store[state_name] = NestedDefaultDict()
        logger.trace(f'State name set -> {state_name}')

    def set_frame(self, name: str, frame: pd.DataFrame) -> None:
        """Sets the frame"""
        if 'original' in name:
            self._original_frame = frame
            logger.trace(f'Original frame set -> {type(frame)}')
        elif 'ephemeral' in name:
            self._ephemeral_frame = frame
            logger.trace(f'Ephemeral frame set -> {type(frame)}')
        else:
            raise ValueError(f'Invalid name -> {name}, allowed -> original, ephemeral')

    def get_frame(self, name: str) -> pd.DataFrame:
        """Returns the frame"""
        if 'original' in name:
            logger.trace(f'Returning original frame -> {type(self._original_frame)}')
            return self._original_frame
        if 'ephemeral' in name:
            logger.trace(f'Returning ephemeral frame -> {type(self._ephemeral_frame)}')
            return self._ephemeral_frame
        raise ValueError(f'Invalid name -> {name}, allowed -> original, ephemeral')

    def set_store(self, name: str, state_name: str, step_name: str, data: pd.DataFrame or list) -> None:
        """Sets the store frame"""
        if 'frame' in name:
            self._frame_store[state_name][step_name] = data
            logger.trace(f'Store data set -> {type(data)}')
        elif 'feature' in name:
            self._feature_store[state_name][step_name] = data
            logger.trace(f'Feature data set -> {type(data)}')
            scores = len(data) * [1]
            scores[: min(10, len(data))] = range(
                10, 10 - min(10, len(data)), -1
            )  # first min(10, len(features)) features get rank score, rest get score of 1
            for i, feature in enumerate(data):  # calculate feature importance scores on the fly
                if feature in self._feature_score_store[step_name].keys():
                    self._feature_score_store[step_name][feature] += scores[i]
                else:
                    self._feature_score_store[step_name][feature] = scores[i]
        elif 'score' in name:
            self._score_store[state_name][step_name] = data
            logger.trace(f'Score data set -> {type(data)}')
        else:
            raise ValueError(f'Invalid data name to set store data -> {name}, allowed -> frame, feature, score')

    def get_store(self, name: str, state_name: str, step_name: str) -> pd.DataFrame:
        """Returns the store value"""
        if name == 'frame':
            logger.trace(f'Returning frame -> {type(self._frame_store[state_name][step_name])}')
            return self._frame_store[state_name][step_name]
        elif name == 'feature':
            logger.trace(f'Returning feature -> {type(self._feature_store[state_name][step_name])}')
            return self._feature_store[state_name][step_name]
        elif name == 'feature_score':
            logger.trace(f'Returning feature scores -> {type(self._feature_score_store[step_name])}')
            return self._feature_score_store[step_name]
        elif name == 'score':
            logger.trace(f'Returning score -> {type(self._score_store[state_name][step_name])}')
            return self._score_store[state_name][step_name]
        raise ValueError(f'Invalid data name to get store data -> {name}, allowed -> frame, feature, score')

    def get_all_features(self) -> dict:
        """Returns the store value"""
        logger.trace(f'Returning all features -> {len(self._feature_store.keys())}')
        return self._feature_store

    def get_feature_job_names(self, state_name: str) -> list:
        """Returns the store value"""
        if state_name not in self._feature_store:
            raise ValueError(f'Invalid state name -> {state_name}')
        logger.trace(f'Returning feature job names -> {type(self._feature_store[state_name])}')
        return list(self._feature_store[state_name].keys())

    def sync_ephemeral_data_to_data_store(self, state_name: str, step_name: str) -> None:
        """Syncs the ephemeral frame with the data store"""
        self._frame_store[state_name][step_name] = self._ephemeral_frame
        logger.trace(f'Ephemeral frame synced -> {type(self._ephemeral_frame)} to data store')

    def remove_state_data_store(self, state_name: str) -> None:
        """Removes the state, prevent memory overflows"""
        if state_name in self._frame_store:
            del self._frame_store[state_name]
            logger.trace(f'State removed -> {state_name}')
        else:
            raise ValueError(f'Invalid state name -> {state_name}')
        
    def save_intermediate_results(self, out_dir) -> None:
        with open(os.path.join(out_dir, 'feature_scores.json'), 'w') as feature_file:
            json.dump(self._feature_score_store, feature_file)
        with open(os.path.join(out_dir, 'scores.json'), 'w') as score_file:
            json.dump(self._score_store, score_file)

    def load_intermediate_results(self, out_dir, opt_scoring):
        with open(os.path.join(out_dir, 'feature_scores.json'), 'r') as feature_file:
            self._feature_score_store = json.load(feature_file)
        with open(os.path.join(out_dir, 'scores.json'), 'r') as score_file:
            self._score_store = json.load(score_file)

        seeds = list(self._score_store.keys())  # use seeds and jobs from results, not from current config
        job_names = list(self._feature_score_store.keys())
        job_names.remove('all_features')
        jobs_n_top = list(self._score_store[seeds[0]].keys())
        models = list(self._score_store[jobs_n_top[0]].keys())
        scores = list(self._score_store[models[0]].keys())
        n_bootstraps = len(scores[opt_scoring])

        return seeds, job_names, models, n_bootstraps