import itertools
import sys
from copy import deepcopy

from dictlib import dig, dug
from loguru import logger
from omegaconf import DictConfig


class StateMachine:
    """Iterates over all possible config states"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config

        self.count = 0
        self.state = None
        self.state_tree = None
        self.max_count = None
        self.state_data = None
        self.state_names = None

        self.create_state_tree()
        self.update_state()

    def __iter__(self) -> object:
        """Return the iterator object"""
        return self

    def __next__(self) -> tuple:
        """Return the next item"""
        self.update_state()
        logger.info(f'State -> {self.state}')
        return self.state, self.get_state_config()

    def create_state_tree(self) -> None:
        """Create a tree of all possible states for the pipeline to run in"""
        self.state_names = [
            'meta.target_label',
            'meta.seed',
            'impute.method',
        ]  # define state names to branch on
        self.state_data = [dig(self.config, x) for x in self.state_names]  # get data for each state name
        self.state_tree = list(itertools.product(*self.state_data))  # create tree of all possible states
        self.max_count = len(self.state_tree)

    def get_state_config(self) -> DictConfig:
        """Return config for current state"""
        config = deepcopy(self.config)
        for state_name, state_value in zip(self.state_names, self.state):
            dug(config, state_name, state_value)  # set state value in config
        return config

    def update_state(self) -> None:
        """Update the current state to the next state"""
        if self.count < self.max_count:
            self.state = self.state_tree[self.count]
            self.count += 1
        else:
            logger.info('All states have been run')
            sys.exit(0)
