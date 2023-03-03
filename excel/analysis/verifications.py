import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from loguru import logger
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from excel.analysis.utils.cross_validation import CrossValidation
from excel.analysis.utils.helpers import init_estimator
from excel.analysis.utils.normalisers import Normaliser


class VerifyFeatures(Normaliser):
    """Train random forest classifier to verify feature importance"""

    def __init__(self, config, v_data, v_data_test=None, features=None, task: str = 'classification'):
        super().__init__()
        self.target_label = config.analysis.experiment.target_label
        self.seed = config.analysis.run.seed
        self.scoring = config.analysis.run.scoring
        self.class_weight = config.analysis.run.class_weight
        self.oversample = config.analysis.run.verification.oversample
        self.param_grids = config.analysis.run.verification.param_grids
        self.task = task
        models_dict = config.analysis.run.verification.models
        self.models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [
            model for model in self.models if 'ensemble' in model
        ]  # separate ensemble model from other models
        self.models = [model for model in self.models if model not in self.ensemble]

        if v_data_test is None:  # v_data needs to be split into train and test set
            x, y = self.prepare_data(v_data, features_to_keep=features)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x,
                y,
                stratify=y,
                test_size=0.20,
                random_state=self.seed,
            )
        else:  # v_data already split into train and test set
            self.x_train, self.y_train = self.prepare_data(v_data, features_to_keep=features)
            self.x_test, self.y_test = self.prepare_data(v_data_test, features_to_keep=features)

        if self.oversample:
            oversampler = RandomOverSampler(random_state=self.seed)
            self.x_train, self.y_train = oversampler.fit_resample(self.x_train, self.y_train)

    def __call__(self):
        """Train random forest classifier to verify feature importance"""
        best_estimators = []
        for model in self.models:
            logger.info(f'Optimising {model} model...')
            param_grid = self.param_grids[model]
            estimator, cross_validator, scoring = init_estimator(
                model, self.task, self.seed, self.scoring, self.class_weight
            )

            optimiser = CrossValidation(
                self.x_train, self.y_train, estimator, cross_validator, param_grid, scoring, self.seed
            )
            best_estimator = optimiser()
            best_estimators.append((model, best_estimator))
            y_pred = best_estimator.predict(self.x_test)
            logger.info(f'Model was optimised using {self.scoring[self.task]}.')
            self.performance_statistics(y_pred)

        for ensemble in self.ensemble:
            logger.info(f'Combining optimised models in {ensemble} estimator')
            if 'voting' in ensemble:
                if self.task == 'classification':
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

    def prepare_data(self, data: pd.DataFrame, features_to_keep: list = None):
        y = data[self.target_label]
        data = self.z_score_norm(data)
        x = data.drop(
            columns=[c for c in data.columns if c not in features_to_keep], axis=1
        )  # Keep only selected features
        if self.target_label in x.columns:  # ensure that target column is dropped
            x = x.drop(self.target_label, axis=1)

        return x, y

    def performance_statistics(self, y_pred):
        if self.task == 'classification':
            print('Accuracy', accuracy_score(self.y_test, y_pred, normalize=True))
            print('Average precision', average_precision_score(self.y_test, y_pred))
            print(classification_report(self.y_test, y_pred))

            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            plt.figure(figsize=(10, 7))
            plt.title('Confusion matrix')
            sns.heatmap(cm, annot=True, fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('Truth')
            # plt.show()
        else:  # regression
            print('Mean absolute error', mean_absolute_error(self.y_test, y_pred))
            print('R2 score', r2_score(self.y_test, y_pred))
            plt.figure(figsize=(10, 7))
            plt.title(f'Regression on {self.target_label}')
            sns.regplot(x=self.y_test, y=y_pred, ci=None)
            plt.xlabel(f'True {self.target_label}')
            plt.ylabel(f'Predicted {self.target_label}')
            plt.show()
