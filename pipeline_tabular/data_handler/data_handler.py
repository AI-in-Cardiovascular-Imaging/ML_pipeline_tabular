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
        '_frame': None,
    }

    def __init__(self) -> None:
        self._frame_store = NestedDefaultDict()
        self._feature_store = NestedDefaultDict()
        self._feature_score_store = NestedDefaultDict()
        self._score_store = NestedDefaultDict()
        self._frame = None
        self.__dict__ = self.shared_state  # borg design pattern

    def set_frame(self,frame: pd.DataFrame) -> None:
        """Sets the frame"""
        self._frame = frame
        logger.trace(f'Frame set -> {type(frame)}')

    def get_frame(self) -> pd.DataFrame:
        """Returns the frame"""
        logger.trace(f'Returning frame -> {type(self._frame)}')
        return self._frame

    def set_store(
        self,
        name: str,
        seed: int,
        job_name: str = None,
        data: pd.DataFrame or list = None,
        boot_iter: int = None,
    ) -> None:
        """Sets the store frame"""
        seed = str(seed)
        boot_iter = str(boot_iter)

        if 'frame' in name:
            self._frame_store[seed][job_name] = data
            logger.trace(f'Store data set -> {type(data)}')
        elif 'feature' in name:
            if seed not in self._feature_store.keys():
                self._feature_store[seed] = NestedDefaultDict()
            if boot_iter not in self._feature_store[seed].keys():  # new boot_iter
                self._feature_store[seed][boot_iter] = NestedDefaultDict()
            self._feature_store[seed][boot_iter][job_name] = data
            logger.trace(f'Feature data set -> {type(data)}')

            if job_name not in self._feature_score_store.keys():
                self._feature_score_store[job_name] = NestedDefaultDict()
            scores = len(data) * [1]
            scores[: min(10, len(data))] = range(
                10, 10 - min(10, len(data)), -1
            )  # first min(10, len(features)) features get rank score, rest get score of 1
            for i, feature in enumerate(data):  # calculate feature importance scores on the fly
                if feature in self._feature_score_store[job_name].keys():
                    self._feature_score_store[job_name][feature] += scores[i]
                else:
                    self._feature_score_store[job_name][feature] = scores[i]
        elif 'score' in name:
            if seed not in self._score_store.keys():
                self._score_store[seed] = {}
            self._score_store[seed][job_name] = data
            logger.trace(f'Score data set -> {type(data)}')
        else:
            raise ValueError(f'Invalid data name to set store data -> {name}, allowed -> frame, feature, score')

    def get_store(self, name: str, seed: int, job_name: str = None, boot_iter: int = None) -> pd.DataFrame:
        """Returns the store value"""
        seed = str(seed)
        boot_iter = str(boot_iter)

        if name == 'frame':
            logger.trace(f'Returning frame -> {type(self._frame_store[seed][job_name])}')
            return self._frame_store[seed][job_name]
        elif name == 'feature':
            logger.trace(f'Returning feature -> {type(self._feature_store[seed][boot_iter][job_name])}')
            return self._feature_store[seed][boot_iter][job_name]
        elif name == 'feature_score':
            logger.trace(f'Returning feature scores -> {type(self._feature_score_store[job_name])}')
            return self._feature_score_store[job_name]
        elif name == 'score':
            try:
                logger.trace(f'Returning score -> {type(self._score_store[seed][job_name])}')
                return self._score_store[seed][job_name]
            except KeyError:
                return {}
        raise ValueError(f'Invalid data name to get store data -> {name}, allowed -> frame, feature, score')

    def save_intermediate_results(self, out_dir) -> None:
        with open(os.path.join(out_dir, 'features.json'), 'w') as feature_file:
            json.dump(self._feature_store, feature_file)
        with open(os.path.join(out_dir, 'feature_scores.json'), 'w') as feature_score_file:
            json.dump(self._feature_score_store, feature_score_file)
        with open(os.path.join(out_dir, 'scores.json'), 'w') as score_file:
            json.dump(self._score_store, score_file)

    def load_intermediate_results(self, out_dir):
        try:
            with open(os.path.join(out_dir, 'features.json'), 'r') as feature_file:
                self._feature_store = json.load(feature_file)
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(out_dir, 'feature_scores.json'), 'r') as feature_score_file:
                self._feature_score_store = json.load(feature_score_file)
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(out_dir, 'scores.json'), 'r') as score_file:
                self._score_store = json.load(score_file)
        except FileNotFoundError:
            return False  # need to init scores nested dict

        return True  # when all files could be read

