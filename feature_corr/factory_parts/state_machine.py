import itertools
from copy import deepcopy

from dictlib import dig, dug
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf


class StateMachine:
    """Iterates over all possible config states"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config

        self.count = 0
        self.state = None
        self.state_tree = None
        self.max_count = None
        self.state_data = None
        self.state_names = [
            'meta.seed',
        ]  # define state names to branch on

        self.check_state_names()
        self.create_state_tree()

    def __iter__(self) -> object:
        """Return the iterator object"""
        return self

    def __next__(self) -> DictConfig:
        """Return the next item"""
        self.update_state()
        logger.info(f'State {self.count}/{self.max_count} -> {self.state[0]}')
        return self.get_state_config()

    def minor_setup(self) -> bool or ListConfig:
        """Minor config setup"""
        if self.config.meta.aggregated_jobs is True:
            return self.config.meta.seed
        if self.config.meta.aggregated_jobs is False:
            return self.config.meta.aggregated_jobs
        if isinstance(self.config.meta.aggregated_jobs, ListConfig):
            return self.config.meta.aggregated_jobs
        raise ValueError('Aggregated seeds must be a True, False or List[int]')

    def check_state_names(self) -> None:
        """Check if all state names have valid types"""
        for state_name in self.state_names:
            if not OmegaConf.is_list(dig(self.config, state_name)):
                raise ValueError(f'Expected variable as list in config -> {state_name}')

    def create_state_tree(self) -> None:
        """Create a tree of all possible states for the pipeline to run in"""
        self.state_data = [dig(self.config, x) for x in self.state_names]  # get data for each state name
        self.state_tree = list(itertools.product(*self.state_data))  # create tree of all possible states
        self.max_count = len(self.state_tree)
        logger.info(f'State machine is about to run -> {self.max_count} states')

    def get_state_config(self) -> DictConfig:
        """Return config for current state"""
        config = deepcopy(self.config)
        for state_name, state_value in zip(self.state_names, self.state):
            dug(config, state_name, state_value)  # set state value in config
        OmegaConf.update(config.meta, 'state_name', '_'.join(map(str, self.state)))
        return config

    def update_state(self) -> None:
        """Update the current state to the next state"""
        if self.count < self.max_count:
            self.state = self.state_tree[self.count]
            self.count += 1
        else:
            logger.info('All states have been run')
            raise StopIteration
