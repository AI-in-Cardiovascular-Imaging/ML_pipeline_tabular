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

    def __init__(self, config: DictConfig, top_feature_names: list = None) -> None:
        super().__init__()
        self.config = config
        self.top_features = top_feature_names
        self.out_dir = os.path.join(config.meta.output_dir, config.meta.name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.seeds = config.meta.seed
        self.workers = config.meta.workers
        self.state_name = config.meta.state_name
        self.learn_task = config.meta.learn_task
        self.target_label = config.meta.target_label
        self.train_scoring = config.selection.scoring
        self.class_weight = config.selection.class_weight
        v_scoring_dict = config.verification.scoring[self.learn_task]
        self.verif_scoring = [v_scoring for v_scoring in v_scoring_dict if v_scoring_dict[v_scoring]]
        models_dict = config.verification.models
        self.param_grids = config.verification.param_grids
        self.models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [model for model in self.models if 'ensemble' in model]  # only ensemble models
        self.models = [model for model in self.models if model not in self.ensemble]
        self.best_estimators = NestedDefaultDict()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def verify_final(self) -> None:
        """Train classifier to verify final feature importance"""
        logger.info('Verifying final feature importance')

        # TODO: maybe there is a better way to do this
        self.config.impute.method = 'simple_impute'
        self.config.data_split.over_sample_method.binary_classification = 'SMOTEN'

        self.feature_sets = [[feature] for feature in self.top_features]
        self.feature_sets.append(list(self.top_features))
        self.feature_names = [feature for feature in self.top_features]
        self.feature_names.append(f'{len(self.top_features)}-item score')
        for seed in self.seeds:
            for top_feature, feature_name in zip(self.feature_sets, self.feature_names):
                self.pre_process_frame(seed)
                self.train_test_split(top_feature)
                self.train_models(seed, feature_name)  # optimise all models
        self.evaluate()  # evaluate all optimised models

    def pre_process_frame(self, seed) -> None:
        """Pre-process frame for verification"""
        frame = self.get_frame('ephemeral')
        TargetStatistics(self.config).verification_mode(frame)
        Imputer(self.config).verification_mode(frame, seed)
        frame = self.get_store('frame', 'verification', 'ephemeral')
        DataSplit(self.config).verification_mode(frame)

    def train_test_split(self, top_feature) -> None:
        """Prepare data for training"""
        v_train = self.get_store('frame', 'verification', 'verification_train')
        v_test = self.get_store('frame', 'verification', 'verification_test')
        self.x_train, self.y_train = self.split_frame(v_train, top_feature)
        self.x_test, self.y_test = self.split_frame(v_test, top_feature)

    def train_models(self, seed, feature_name) -> None:
        """Train classifier to verify feature importance"""
        estimators = []
        for model in self.models:
            param_grid = self.param_grids[model]
            estimator, cross_validator, scoring = init_estimator(
                model,
                self.learn_task,
                seed,
                self.train_scoring,
                self.class_weight,
            )
            optimiser = CrossValidation(
                self.x_train,
                self.y_train,
                estimator,
                cross_validator,
                param_grid,
                scoring,
                seed,
                self.workers,
            )
            best_estimator = optimiser()
            estimators.append((model, best_estimator))
            self.best_estimators[model][seed][feature_name] = best_estimator  # store for evaluation later

        for ensemble in self.ensemble:
            if 'voting' in ensemble:
                if self.learn_task == 'binary_classification':
                    ens_estimator = VotingClassifier(estimators=estimators, voting='soft')
                    ens_estimator.estimators_ = [
                        est_tuple[1] for est_tuple in estimators
                    ]  # best_estimators are already fit -> need to set estimators_, le_ and classes_
                    ens_estimator.le_ = LabelEncoder().fit(self.y_test)
                    ens_estimator.classes_ = ens_estimator.le_.classes_
                else:  # regression
                    ens_estimator = VotingRegressor(estimators=estimators)
                    ens_estimator.estimators_ = [
                        est_tuple[1] for est_tuple in estimators
                    ]  # best_estimators are already fit -> need to set estimators_
            else:
                logger.error(f'{ensemble} has not yet been implemented.')
                raise NotImplementedError

            self.best_estimators[ensemble][seed][feature_name] = ens_estimator  # store for evaluation later

    def evaluate(self) -> None:
        """Evaluate all optimised models"""
        fig_roc_models, ax_roc_models = plt.subplots()
        fig_prc_models, ax_prc_models = plt.subplots()
        for model in self.models + self.ensemble:
            fig_roc, ax_roc = plt.subplots()
            fig_prc, ax_prc = plt.subplots()
            for top_feature, feature_name in zip(self.feature_sets, self.feature_names):
                scores = {score: [] for score in self.verif_scoring}
                tprs = []
                precisions = []
                mean_x = np.linspace(0, 1, 100)
                for seed in self.seeds:
                    estimator = self.best_estimators[model][seed][feature_name]
                    y_pred = estimator.predict(self.x_test[top_feature])
                    probas, fpr, tpr, precision, recall = self.auc(estimator, top_feature)
                    for score in self.verif_scoring:  # calculate and store all requested scores
                        try:
                            scores[score].append(getattr(metrics, score)(self.y_test, probas))
                        except ValueError:
                            scores[score].append(getattr(metrics, score)(self.y_test, y_pred))
                    interp_tpr = np.interp(mean_x, fpr, tpr)  # AUROC
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    interp_recall = np.interp(mean_x, precision, recall)  # AUPRC
                    interp_recall[0] = 1.0
                    precisions.append(interp_recall)
                scores = {
                    score: f'{np.mean(scores[score]):.3f} +- {np.std(scores[score]):.3f}'
                    for score in self.verif_scoring
                }  # compute mean +- std for all scores
                mean_tpr = np.mean(tprs, axis=0)  # compute mean +- std for AUC plots
                ax_roc.plot(mean_x, mean_tpr, label=f'{feature_name}, AUROC={scores["roc_auc_score"]}', alpha=0.7)
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                ax_roc.fill_between(mean_x, tprs_lower, tprs_upper, color='grey', alpha=0.2)
                mean_precision = np.mean(precisions, axis=0)  # AUPRC
                ax_prc.plot(
                    mean_x,
                    mean_precision,
                    label=f'{feature_name}, AUPRC={scores["average_precision_score"]}',
                    alpha=0.7,
                )
                std_precision = np.std(precisions, axis=0)
                precisions_upper = np.minimum(mean_precision + std_precision, 1)
                precisions_lower = np.maximum(mean_precision - std_precision, 0)
                ax_prc.fill_between(mean_x, precisions_lower, precisions_upper, color='grey', alpha=0.2)

            if y_pred.sum() == 0:
                logger.warning(
                    f'0/{int(self.y_test.sum())} positive samples were predicted using top features {top_feature}.'
                )
            self.save_plots(fig_roc, ax_roc, fig_prc, ax_prc, f'{model}.pdf')  # store and clear model-wise plots
            self.performance_statistics(scores, y_pred)
            ax_roc_models.plot(mean_x, mean_tpr, label=f'{model}, AUROC={scores["roc_auc_score"]}', alpha=0.7)
            ax_roc_models.fill_between(mean_x, tprs_lower, tprs_upper, color='grey', alpha=0.2)
            ax_prc_models.plot(
                mean_x, mean_precision, label=f'{model}, AUPRC={scores["average_precision_score"]}', alpha=0.7
            )
            ax_prc_models.fill_between(mean_x, precisions_lower, precisions_upper, color='grey', alpha=0.2)

        self.save_plots(fig_roc_models, ax_roc_models, fig_prc_models, ax_prc_models, 'all_models.pdf')

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

    def save_plots(self, fig_roc, ax_roc, fig_prc, ax_prc, name: str) -> None:
        ax_roc.set_title('Receiver-operator curve (ROC)')
        ax_roc.set_xlabel('1 - Specificity')
        ax_roc.set_ylabel('Sensitivity')
        ax_roc.grid()
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Baseline, AUROC=0.5', alpha=0.7)  # baseline
        ax_roc.legend()
        fig_roc.savefig(os.path.join(self.out_dir, f'AUROC_{name}'))
        fig_roc.clear()
        ax_prc.set_title('Precision-recall curve (PRC)')
        ax_prc.set_xlabel('Recall (Sensitivity)')
        ax_prc.set_ylabel('Precision')
        ax_prc.grid()
        pos_rate = round(self.y_test.sum() / len(self.y_test), 3)
        ax_prc.axhline(
            y=pos_rate, color='k', linestyle='--', label=f'Baseline, AUPRC={pos_rate}', alpha=0.7
        )  # baseline
        ax_prc.legend()
        fig_prc.savefig(os.path.join(self.out_dir, f'AUPRC_{name}'))
        fig_prc.clear()

    def split_frame(self, frame: pd.DataFrame, top_features) -> tuple:
        """Prepare frame for verification"""
        frame, _ = self.z_score_norm(frame)  # todo: config
        y_frame = frame[self.target_label]
        x_frame = frame[top_features]  # only keep top features
        if self.target_label in x_frame.columns:
            x_frame = x_frame.drop(self.target_label, axis=1)
            logger.warning(f'{self.target_label} was found in the top features for validation')
        return x_frame, y_frame
