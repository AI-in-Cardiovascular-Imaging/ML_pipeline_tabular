import os
import shap

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import imblearn.metrics as imb_metrics
from loguru import logger
from omegaconf import DictConfig
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from pipeline_tabular.utils.helpers import init_estimator
from pipeline_tabular.utils.normalisers import Normalisers
from pipeline_tabular.data_handler.data_handler import DataHandler, NestedDefaultDict


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


class Verification(DataHandler, Normalisers):
    """Train random forest classifier to verify feature importance"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.workers = config.meta.workers
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.train_scoring = config.selection.scoring
        self.class_weight = config.selection.class_weight
        self.n_top_features = config.verification.use_n_top_features
        v_scoring_dict = config.verification.scoring[self.learn_task]
        self.verif_scoring = [v_scoring for v_scoring in v_scoring_dict if v_scoring_dict[v_scoring]]
        models_dict = config.verification.models
        self.param_grids = config.verification.param_grids
        self.models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [model for model in self.models if 'ensemble' in model]  # only ensemble models
        self.models = [model for model in self.models if model not in self.ensemble]
        if len(self.models) < 2:  # ensemble methods need at least two models to combine their results
            self.ensemble = []
        self.best_estimators = NestedDefaultDict()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_test_raw = None

    def __call__(self, seed, boot_iter, job_name, imputer, model=None, n_top_features=None, explain_mode=False):
        """Train classifier to verify final feature importance"""
        self.seed = seed
        self.boot_iter = boot_iter
        self.imputer = imputer
        self.models = model if model is not None else self.models  # single model provided by Explain class
        self.explain_mode = explain_mode

        self.train_test_split()
        top_features = self.get_store('feature', seed, job_name, boot_iter)
        if not explain_mode:
            n_top_features = [n for n in self.n_top_features if n <= len(top_features)]
        for n_top in n_top_features:
            logger.info(f'Verifying final feature importance for top {n_top} features...')
            self.top_features = top_features[:n_top]
            self.train_models(f'{job_name}_{n_top}')  # optimise all models
            pred_function, conf_matrix = self.evaluate(f'{job_name}_{n_top}')  # evaluate all optimised models

        return pred_function, conf_matrix, self.x_train, self.x_test  # only needed for Explain class

    def train_test_split(self) -> None:
        """Prepare data for training"""
        v_train = self.get_store('frame', self.seed, 'train')
        v_test = self.get_store('frame', self.seed, 'test')
        self.x_train, self.y_train, _ = self.split_frame(v_train)
        self.x_test, self.y_test, self.x_test_raw = self.split_frame(
            v_test, normalise=True
        )  # test data not yet normalised

    def train_models(self, job_name) -> None:
        """Train classifier to verify feature importance"""
        estimators = []
        for model in self.models:
            try:
                scores = self.get_store('score', self.seed, job_name)[model]
            except KeyError:  # model not yet stored for this seed/job
                scores = {scoring: [] for scoring in self.verif_scoring}
            if len(scores[self.verif_scoring[0]]) < self.boot_iter + 1 or self.explain_mode:  # bootstraps missing
                logger.info(f'Training {model} model...')
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
                    self.x_train[self.top_features],
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
                self.best_estimators[model] = best_estimator  # store for evaluation later

        for ensemble in self.ensemble:
            logger.info(f'Training {ensemble} ensemble model...')
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

            self.best_estimators[ensemble] = ens_estimator  # store for evaluation later

    def evaluate(self, job_name):
        """Evaluate all optimised models"""
        # pred_func = None
        scores = self.get_store('score', self.seed, job_name)
        models = self.models if self.explain_mode else (self.models + self.ensemble)
        for i, model in enumerate(models):
            if model not in scores.keys():
                scores[model] = {scoring: [] for scoring in self.verif_scoring + ['probas', 'true', 'pred', 'pos_rate']}
            if len(scores[model][self.verif_scoring[0]]) < self.boot_iter + 1 or self.explain_mode:
                logger.info(f'Evaluating {model} model ({i+1}/{len(models)})...')
                estimator = self.best_estimators[model]
                y_pred = estimator.predict(self.x_test[self.top_features])
                pred_func, probas = self.get_predictions(estimator)
                if self.explain_mode:
                    conf_matrix = metrics.confusion_matrix(self.y_test, y_pred, labels=estimator.classes_)
                    conf_matrix = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=estimator.classes_)
                    return pred_func, conf_matrix
                for score in self.verif_scoring:  # calculate and store all requested scores
                    if score not in scores[model].keys():
                        scores[model][score] = []
                    try:
                        scores[model][score].append(getattr(metrics, score)(self.y_test, probas))
                    except ValueError:
                        scores[model][score].append(getattr(metrics, score)(self.y_test, y_pred))
                    except AttributeError:  # try imbalanced learn metrics (e.g. for specificity)
                        try:
                            scores[model][score].append(getattr(imb_metrics, score)(self.y_test, probas))
                        except ValueError:
                            scores[model][score].append(getattr(imb_metrics, score)(self.y_test, y_pred))
                            
                scores[model]['probas'].append(probas.tolist())  # need list to save as json later
                scores[model]['pred'].append(y_pred.tolist())
                scores[model]['true'].append(self.y_test.tolist())
                scores[model]['pos_rate'].append(round(self.y_test.sum() / len(self.y_test), 3))
                if y_pred.sum() == 0:
                    logger.warning(f'0/{int(self.y_test.sum())} positive samples were predicted using top features.')
        self.set_store('score', self.seed, job_name, scores)  # store results for summary in report

        return None
        
    def get_predictions(self, best_estimator):
        try:  # e.g. for logistic regression
            probas = best_estimator.decision_function(self.x_test[self.top_features])
        except AttributeError:  # other estimators
            probas = best_estimator.predict_proba(self.x_test[self.top_features])[:, 1]

        pred_func = best_estimator.predict_proba
        return pred_func, probas

    def split_frame(self, frame: pd.DataFrame, normalise=False) -> tuple:
        """Prepare frame for verification"""
        x_frame_raw = None
        if normalise:
            x_frame_raw = frame.drop(self.target_label, axis=1)
            frame = self.normalise_test(frame)
        y_frame = frame[self.target_label]
        x_frame = frame.drop(self.target_label, axis=1)
        return x_frame, y_frame, x_frame_raw

    def normalise_test(self, frame: pd.DataFrame) -> pd.DataFrame:
        non_categorical = self.scaler.feature_names_in_  # use same features as in train fit
        to_normalise = frame[non_categorical]
        tmp_label = frame[self.target_label]  # keep label col as is
        try:
            arr_frame = to_normalise.drop(self.target_label, axis=1).values
        except KeyError:  # target label is categorical -> already removed
            arr_frame = to_normalise.values
        norm_frame = self.scaler.transform(arr_frame)
        frame[non_categorical] = norm_frame
        frame[self.target_label] = tmp_label
        return frame
