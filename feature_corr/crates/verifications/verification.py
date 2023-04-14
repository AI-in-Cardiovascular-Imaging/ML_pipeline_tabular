import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from loguru import logger
from omegaconf import DictConfig
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from feature_corr.crates.data_split import DataSplit
from feature_corr.crates.helpers import init_estimator
from feature_corr.crates.imputers import Imputer
from feature_corr.crates.inspections import TargetStatistics
from feature_corr.crates.normalisers import Normalisers
from feature_corr.data_borg import DataBorg, NestedDefaultDict


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

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.out_dir = os.path.join(config.meta.output_dir, config.meta.name, str(config.meta.seed))
        os.makedirs(self.out_dir, exist_ok=True)
        self.seed = config.meta.seed
        self.workers = config.meta.workers
        self.state_name = config.meta.state_name
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.jobs = config.selection.jobs
        self.train_scoring = config.selection.scoring
        self.class_weight = config.selection.class_weight
        self.n_top_features = self.config.verification.use_n_top_features
        v_scoring_dict = config.verification.scoring[self.learn_task]
        self.verif_scoring = [v_scoring for v_scoring in v_scoring_dict if v_scoring_dict[v_scoring]]
        models_dict = config.verification.models
        self.param_grids = config.verification.param_grids
        self.models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [model for model in self.models if 'ensemble' in model]  # only ensemble models
        self.models = [model for model in self.models if model not in self.ensemble]
        if len(self.models) < 2:  # ensemble methods need at least two models two combine their results
            self.ensemble = []
        self.best_estimators = NestedDefaultDict()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def __call__(self, job_name) -> None:
        """Train classifier to verify final feature importance"""
        logger.info('Verifying final feature importance')

        # TODO: maybe there is a better way to do this
        self.config.impute.method = 'simple_impute'
        self.config.data_split.over_sample_method.binary_classification = 'SMOTEN'

        self.top_features = self.get_store('feature', str(self.seed), job_name)[: self.n_top_features]
        self.feature_sets = [[feature] for feature in self.top_features]
        self.feature_sets.append(list(self.top_features))
        self.feature_names = [feature for feature in self.top_features]
        self.feature_names.append(f'{len(self.top_features)}-item score')
        for top_feature, feature_name in zip(self.feature_sets, self.feature_names):
            self.pre_process_frame()
            self.train_test_split(top_feature)
            self.train_models(feature_name)  # optimise all models
        self.evaluate(job_name)  # evaluate all optimised models

    def pre_process_frame(self) -> None:
        """Pre-process frame for verification"""
        frame = self.get_frame('ephemeral')
        TargetStatistics(self.config).verification_mode(frame)
        Imputer(self.config).verification_mode(frame, self.seed)
        frame = self.get_store('frame', 'verification', 'ephemeral')
        DataSplit(self.config).verification_mode(frame)

    def train_test_split(self, top_feature) -> None:
        """Prepare data for training"""
        v_train = self.get_store('frame', 'verification', 'verification_train')
        v_test = self.get_store('frame', 'verification', 'verification_test')
        self.x_train, self.y_train = self.split_frame(v_train, top_feature)
        self.x_test, self.y_test = self.split_frame(v_test, top_feature)

    def train_models(self, feature_name) -> None:
        """Train classifier to verify feature importance"""
        estimators = []
        for model in self.models:
            param_grid = self.param_grids[model]
            estimator, cross_validator, scoring = init_estimator(
                model,
                self.learn_task,
                self.seed,
                self.train_scoring,
                self.class_weight,
                self.workers,
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
            estimators.append((model, best_estimator))
            self.best_estimators[model][feature_name] = best_estimator  # store for evaluation later

        for ensemble in self.ensemble:
            if 'voting' in ensemble:
                if self.learn_task == 'binary_classification':
                    ens_estimator = VotingClassifier(estimators=estimators, voting='soft', n_jobs=self.workers)
                    ens_estimator.estimators_ = [
                        est_tuple[1] for est_tuple in estimators
                    ]  # best_estimators are already fit -> need to set estimators_, le_ and classes_
                    ens_estimator.le_ = LabelEncoder().fit(self.y_test)
                    ens_estimator.classes_ = ens_estimator.le_.classes_
                else:  # regression
                    ens_estimator = VotingRegressor(estimators=estimators, n_jobs=self.workers)
                    ens_estimator.estimators_ = [
                        est_tuple[1] for est_tuple in estimators
                    ]  # best_estimators are already fit -> need to set estimators_
            else:
                logger.error(f'{ensemble} has not yet been implemented.')
                raise NotImplementedError

            self.best_estimators[ensemble][feature_name] = ens_estimator  # store for evaluation later

    def evaluate(self, job_name) -> None:
        """Evaluate all optimised models"""
        scores = NestedDefaultDict()
        for i, model in enumerate(self.models + self.ensemble):
            logger.info(f'Evaluating {model} model ({i+1}/{len(self.models + self.ensemble)})...')
            for top_feature, feature_name in zip(self.feature_sets, self.feature_names):
                scores[model][feature_name] = {}
                estimator = self.best_estimators[model][feature_name]
                y_pred = estimator.predict(self.x_test[top_feature])
                probas, fpr, tpr, precision, recall = self.auc(estimator, top_feature)
                for score in self.verif_scoring:  # calculate and store all requested scores
                    try:
                        scores[model][top_feature][score] = getattr(metrics, score)(self.y_test, probas)
                    except ValueError:
                        scores[model][top_feature][score] = getattr(metrics, score)(self.y_test, y_pred)

                scores[model][feature_name]['fpr'] = fpr
                scores[model][feature_name]['tpr'] = tpr
                scores[model][feature_name]['precision'] = precision
                scores[model][feature_name]['recall'] = recall
                scores[model][feature_name]['pos_rate'] = round(self.y_test.sum() / len(self.y_test), 3)

            if y_pred.sum() == 0:
                logger.warning(
                    f'0/{int(self.y_test.sum())} positive samples were predicted using top features {top_feature}.'
                )

        self.set_store('score', str(self.seed), job_name)  # store results for summary in report

    def performance_statistics(self, scores, y_pred: pd.DataFrame) -> None:
        """Print performance statistics"""
        if self.learn_task == 'binary_classification':
            print('Averaged metrics:')
            for metric, score in scores.items():
                print(f'{metric}: {score}')

            print('Metrics calculated from single evaluation:')
            print(metrics.classification_report(self.y_test, y_pred, zero_division=0))
            conf_m = metrics.confusion_matrix(self.y_test, y_pred)
            print(conf_m)
            plt.figure(figsize=(10, 7))
            plt.title('Confusion matrix')
            sns.heatmap(conf_m, annot=True, fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('Truth')
            # plt.show()
        elif self.learn_task == 'multi_classification':
            raise NotImplementedError('Multi-classification has not yet been implemented.')
        elif self.learn_task == 'regression':
            print('Mean absolute error', metrics.mean_absolute_error(self.y_test, y_pred))
            print('R2 score', metrics.r2_score(self.y_test, y_pred))
            plt.figure(figsize=(10, 7))
            plt.title(f'Regression on {self.target_label}')
            sns.regplot(x=self.y_test, y=y_pred, ci=None)
            plt.xlabel(f'True {self.target_label}')
            plt.ylabel(f'Predicted {self.target_label}')
            # plt.show()
        else:
            NotImplementedError(f'{self.learn_task} has not yet been implemented.')

    def auc(self, best_estimator, top_feature):
        try:  # for logistic regression
            probas = best_estimator.decision_function(self.x_test[top_feature])
        except AttributeError:  # other estimators
            probas = best_estimator.predict_proba(self.x_test[top_feature])[:, 1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test, probas, drop_intermediate=True)  # AUROC
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, probas)  # AUPRC

        return probas, fpr, tpr, precision, recall

    def split_frame(self, frame: pd.DataFrame, top_features) -> tuple:
        """Prepare frame for verification"""
        frame, _ = self.z_score_norm(frame)  # todo: config
        y_frame = frame[self.target_label]
        x_frame = frame[top_features]  # only keep top features
        if self.target_label in x_frame.columns:
            x_frame = x_frame.drop(self.target_label, axis=1)
            logger.warning(f'{self.target_label} was found in the top features for validation')
        return x_frame, y_frame
