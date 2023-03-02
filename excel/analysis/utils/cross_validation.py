from loguru import logger
from sklearn.model_selection import GridSearchCV


class CrossValidation:
    def __init__(self, x_train, y_train, estimator, cross_validator, param_grid: dict, scoring: str, seed: int) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.param_grid = dict(param_grid)
        self.scoring = scoring
        self.seed = seed

    def __call__(self):
        selector = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cross_validator,
        )
        selector.fit(self.x_train, self.y_train)

        return selector

    # TODO: voting classifier cv