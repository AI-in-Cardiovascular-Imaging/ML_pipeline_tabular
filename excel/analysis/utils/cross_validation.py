from loguru import logger
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, StratifiedKFold


class CrossValidation:
    def __init__(self, x_train, y_train, estimator, param_grid: dict, scoring: str, seed: int) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.estimator = estimator
        self.param_grid = dict(param_grid)
        self.scoring = scoring
        self.seed = seed

    def __call__(self):
        cv = StratifiedKFold(shuffle=True, random_state=self.seed)
        selector = HalvingGridSearchCV(
            estimator=self.estimator, param_grid=self.param_grid, scoring=self.scoring, cv=cv
        )
        selector.fit(self.x_train, self.y_train)

        return selector
