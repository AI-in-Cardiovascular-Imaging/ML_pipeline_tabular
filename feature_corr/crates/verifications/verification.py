import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from omegaconf import DictConfig
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from feature_corr.crates.data_split import DataSplit
from feature_corr.crates.helpers import init_estimator
from feature_corr.crates.imputers import Imputer
from feature_corr.crates.inspections import TargetStatistics
from feature_corr.crates.normalisers import Normalisers
from feature_corr.data_borg import DataBorg


class CrossValidation:
    """Cross validation for feature selection"""

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        estimator: str,
        cross_validator: str,
        param_grid: dict,
        scoring: str,
        seed: int,
        workers: int,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.param_grid = dict(param_grid)
        self.scoring = scoring
        self.seed = seed
        self.workers = workers

    def __call__(self):
        selector = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cross_validator,
            n_jobs=self.workers,
        )
        selector.fit(self.x_train, self.y_train)
        return selector


class Verification(DataBorg, Normalisers):
    """Train random forest classifier to verify feature importance"""

    def __init__(self, config: DictConfig, top_feature_names: list = None) -> None:
        super().__init__()
        self.config = config
        self.top_feature_names = top_feature_names
        self.seed = config.meta.seed
        self.workers = config.meta.workers
        self.state_name = config.meta.state_name
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.scoring = config.selection.scoring
        self.class_weight = config.selection.class_weight
        self.param_grids = config.verification.param_grids
        models_dict = config.verification.models
        self.models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [model for model in self.models if 'ensemble' in model]  # only ensemble models
        self.models = [model for model in self.models if model not in self.ensemble]
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def verify_jobs(self):
        """Verify feature importance per job"""
        jobs = self.get_feature_job_names(self.state_name)
        for job in jobs:
            v_train = self.get_store('frame', self.state_name, 'verification_train')
            v_test = self.get_store('frame', self.state_name, 'verification_test')
            self.top_feature_names = self.get_store('feature', self.state_name, job)
            self.x_train, self.y_train = self.split_frame(v_train)
            self.x_test, self.y_test = self.split_frame(v_test)
            self.train_models()

    def verify_final(self) -> None:
        """Train random forest classifier to verify final feature importance"""
        logger.info('Verifying final feature importance')
        self.pre_process_frame()
        self.train_test_split()
        self.train_models()

    def pre_process_frame(self) -> None:
        """Pre-process frame for verification"""
        self.config.impute.method = 'simple_impute'
        frame = self.get_frame('ephemeral')
        TargetStatistics(self.config).verification_mode(frame)
        frame = self.get_store('frame', 'verification', 'ephemeral')
        Imputer(self.config).verification_mode(frame, self.seed)
        frame = self.get_store('frame', 'verification', 'ephemeral')
        DataSplit(self.config).verification_mode(frame, self.seed)

    def train_test_split(self) -> None:
        """Prepare data for training"""
        v_train = self.get_store('frame', 'verification', 'verification_train')
        v_test = self.get_store('frame', 'verification', 'verification_test')
        self.x_train, self.y_train = self.split_frame(v_train)
        self.x_test, self.y_test = self.split_frame(v_test)

    def train_models(self) -> None:
        """Train random forest classifier to verify feature importance"""
        best_estimators = []
        for model in self.models:
            logger.info(f'Optimising {model} model...')
            param_grid = self.param_grids[model]
            estimator, cross_validator, scoring = init_estimator(
                model,
                self.learn_task,
                self.seed,
                self.scoring,
                self.class_weight,
            )

            optimiser = CrossValidation(
                self.x_train,
                self.y_train,
                estimator,
                cross_validator,
                param_grid,
                scoring,
                self.seed,
                self.workers,
            )
            best_estimator = optimiser()
            best_estimators.append((model, best_estimator))
            y_pred = best_estimator.predict(self.x_test)
            logger.info(f'Model was optimised using {self.scoring[self.learn_task]}.')
            self.performance_statistics(y_pred)

        for ensemble in self.ensemble:
            logger.info(f'Combining optimised models in {ensemble} estimator')
            if 'voting' in ensemble:
                if self.learn_task == 'binary_classification':
                    ens_estimator = VotingClassifier(estimators=best_estimators, voting='hard')
                    ens_estimator.estimators_ = [
                        est_tuple[1] for est_tuple in best_estimators
                    ]  # best_estimators are already fit -> need to set estimators_, le_ and classes_
                    ens_estimator.le_ = LabelEncoder().fit(self.y_test)
                    ens_estimator.classes_ = ens_estimator.le_.classes_
                else:  # regression
                    ens_estimator = VotingRegressor(estimators=best_estimators)
                    ens_estimator.estimators_ = [
                        est_tuple[1] for est_tuple in best_estimators
                    ]  # best_estimators are already fit -> need to set estimators_
            else:
                logger.error(f'{ensemble} has not yet been implemented.')
                raise NotImplementedError

            y_pred = ens_estimator.predict(self.x_test)
            self.performance_statistics(y_pred)

    def performance_statistics(self, y_pred: pd.DataFrame) -> None:
        """Print performance statistics"""
        if self.learn_task == 'binary_classification':
            print('Accuracy', accuracy_score(self.y_test, y_pred, normalize=True))
            print('Average precision', average_precision_score(self.y_test, y_pred))
            print(classification_report(self.y_test, y_pred))
            conf_m = confusion_matrix(self.y_test, y_pred)
            print(conf_m)
            plt.figure(figsize=(10, 7))
            plt.title('Confusion matrix')
            sns.heatmap(conf_m, annot=True, fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('Truth')
            plt.show()
        elif self.learn_task == 'multi_classification':
            raise NotImplementedError('Multi-classification has not yet been implemented.')
        elif self.learn_task == 'regression':
            print('Mean absolute error', mean_absolute_error(self.y_test, y_pred))
            print('R2 score', r2_score(self.y_test, y_pred))
            plt.figure(figsize=(10, 7))
            plt.title(f'Regression on {self.target_label}')
            sns.regplot(x=self.y_test, y=y_pred, ci=None)
            plt.xlabel(f'True {self.target_label}')
            plt.ylabel(f'Predicted {self.target_label}')
            plt.show()
        else:
            NotImplementedError(f'{self.learn_task} has not yet been implemented.')

    def split_frame(self, frame: pd.DataFrame) -> tuple:
        """Prepare frame for verification"""
        y_frame = frame[self.target_label]
        x_frame = frame[self.top_feature_names]  # only keep top features
        if self.target_label in x_frame.columns:
            x_frame = x_frame.drop(self.target_label, axis=1)
            logger.warning(f'{self.target_label} was found in the top features for validation')
        return x_frame, y_frame
