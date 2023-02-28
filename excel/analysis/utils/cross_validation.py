from loguru import logger
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV


class CrossValidation:
    def __init__(self, model: str, param_grid: dict) -> None:
        self.model = model
        self.param_grid = param_grid

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass