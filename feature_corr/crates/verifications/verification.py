import os

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
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
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

        self.fig_roc, self.ax_roc = plt.subplots()
        self.fig_prc, self.ax_prc = plt.subplots()

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
        for self.seed in self.seeds:
            for self.top_features, self.feature_name in zip(self.feature_sets, self.feature_names):
                self.pre_process_frame()
                self.train_test_split()
                self.train_models()  # optimise all models
        self.evaluate()  # evaluate all optimised models

    def pre_process_frame(self) -> None:
        """Pre-process frame for verification"""
        frame = self.get_frame('ephemeral')
        TargetStatistics(self.config).verification_mode(frame)
        Imputer(self.config).verification_mode(frame, self.seed)
        frame = self.get_store('frame', 'verification', 'ephemeral')
        DataSplit(self.config).verification_mode(frame)

    def train_test_split(self) -> None:
        """Prepare data for training"""
        v_train = self.get_store('frame', 'verification', 'verification_train')
        v_test = self.get_store('frame', 'verification', 'verification_test')
        self.x_train, self.y_train = self.split_frame(v_train)
        self.x_test, self.y_test = self.split_frame(v_test)

    def train_models(self) -> None:
        """Train classifier to verify feature importance"""
        estimators = []
        for model in self.models:
            logger.info(f'Optimising {model} model...')
            param_grid = self.param_grids[model]
            estimator, cross_validator, scoring = init_estimator(
                model,
                self.learn_task,
                self.seed,
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
                self.seed,
                self.workers,
            )
            best_estimator = optimiser()
            estimators.append((model, best_estimator))
            self.best_estimators[model][self.seed][self.feature_name] = best_estimator  # store for evaluation later
            logger.info(f'Model was optimised using {self.train_scoring[self.learn_task]}.')
            y_pred = best_estimator.predict(self.x_test)
            self.auc_plots(y_pred, best_estimator)
            self.performance_statistics(y_pred)

        for ensemble in self.ensemble:
            logger.info(f'Combining optimised models in {ensemble} estimator')
            if 'voting' in ensemble:
                if self.learn_task == 'binary_classification':
                    ens_estimator = VotingClassifier(estimators=estimators, voting='hard')
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
            self.best_estimators[ens_estimator][self.seed][
                self.feature_name
            ] = best_estimator  # store for evaluation later

    def evaluate(self) -> None:
        """Evaluate all optimised models"""
        fig_all, ax_all = plt.subplots()
        for model in self.models + self.ensemble:
            for feature_name in self.feature_names:
                # score collection init for each metric according to config dict
                for seed in self.seeds:
                    estimator = self.best_estimators[model][seed][feature_name]
                    y_pred = estimator.predict(self.x_test)
                    # automatically call sklearn function according to config dict, store in collection

                # compute mean + sd
                # compute AUC mean + sd
                # plots, etc.

            # store and clear model-wise plots
            # plot on ax_all
            # (opt.) print some results

        # store fig_all
        # (opt.) print some summary results

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
            # plt.show()
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

    def auc_plots(self, y_pred: pd.DataFrame, best_estimator) -> None:
        try:  # for logistic regression
            probas = best_estimator.decision_function(self.x_test)
        except AttributeError:
            probas = best_estimator.predict_proba(self.x_test)[:, 1]

        if len(self.top_features) == 1:  # name clean-up
            f_name = self.top_features[0]
        else:
            f_name = f'{len(self.top_features)}-item score'
        fpr, tpr, _ = roc_curve(self.y_test, probas, drop_intermediate=True)  # AUROC
        auroc = round(roc_auc_score(self.y_test, y_pred), 3)
        self.ax_roc.plot(fpr, tpr, label=f'{f_name}, AUROC={str(auroc)}', alpha=0.7)
        precision, recall, _ = precision_recall_curve(self.y_test, probas)  # AUPRC
        auprc = round(average_precision_score(self.y_test, y_pred), 3)
        self.ax_prc.plot(precision, recall, label=f'{f_name}, AUPRC={str(auprc)}', alpha=0.7)

    def save_plots(self) -> None:
        self.ax_roc.set_title('Receiver-operator curve (ROC)')
        self.ax_roc.set_xlabel('1 - Specificity')
        self.ax_roc.set_ylabel('Sensitivity')
        self.ax_roc.grid()
        self.ax_roc.plot([0, 1], [0, 1], 'k--', label='Baseline, AUROC=0.5', alpha=0.7)  # baseline
        self.ax_roc.legend()
        self.fig_roc.savefig(os.path.join(self.out_dir, f'AUROC_{self.model}.pdf'))
        self.fig_roc.clear()
        self.ax_prc.set_title('Precision-recall curve (PRC)')
        self.ax_prc.set_xlabel('Recall (Sensitivity)')
        self.ax_prc.set_ylabel('Precision')
        self.ax_prc.grid()
        pos_rate = round(self.y_test.sum() / len(self.y_test), 3)
        self.ax_prc.axhline(
            y=pos_rate, color='k', linestyle='--', label=f'Baseline, AUPRC={pos_rate}', alpha=0.7
        )  # baseline
        self.ax_prc.legend()
        self.fig_prc.savefig(os.path.join(self.out_dir, f'AUPRC_{self.model}.pdf'))
        self.fig_prc.clear()

    def split_frame(self, frame: pd.DataFrame) -> tuple:
        """Prepare frame for verification"""
        frame, _ = self.z_score_norm(frame)  # todo: config
        y_frame = frame[self.target_label]
        x_frame = frame[self.top_features]  # only keep top features
        if self.target_label in x_frame.columns:
            x_frame = x_frame.drop(self.target_label, axis=1)
            logger.warning(f'{self.target_label} was found in the top features for validation')
        return x_frame, y_frame
