import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import rcParams
from omegaconf import DictConfig

from feature_corr.crates.helpers import job_name_cleaner
from feature_corr.data_borg import DataBorg, NestedDefaultDict


class Report(DataBorg):
    """What's my purpose? You are passing the features"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        experiment_name = config.meta.name
        self.seeds = config.meta.seed
        self.output_dir = os.path.join(config.meta.output_dir, experiment_name)
        self.feature_file_path = os.path.join(self.output_dir, 'all_features.json')
        self.jobs = config.selection.jobs
        self.job_names = job_name_cleaner(self.jobs)
        self.n_top_features = self.config.verification.use_n_top_features
        models_dict = config.verification.models
        self.models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [model for model in self.models if 'ensemble' in model]  # only ensemble models
        self.models = [model for model in self.models if model not in self.ensemble]
        if len(self.models) < 2:  # ensemble methods need at least two models two combine their results
            self.ensemble = []
        self.all_features = None

    def __call__(self):
        """Run feature report"""
        all_features = self.get_all_features()
        if all_features:
            self.write_to_file(all_features)
            self.summarise_selection(all_features)
            self.summarise_verification()
        else:
            logger.warning('No features found to report')

    def write_to_file(self, all_features: dict) -> None:
        """Write features to file"""
        with open(self.feature_file_path, 'w+', encoding='utf-8') as file:
            json.dump(all_features, file, indent=4)

        with open(self.feature_file_path, 'r', encoding='utf-8') as file:
            loaded_features = json.load(file)

        if loaded_features != all_features:
            logger.warning(f'Failed to write features -> {self.feature_file_path}')
        else:
            logger.info(f'Saved features to -> {self.feature_file_path}')

    def load_features(self) -> None:
        """Load features from file"""
        if not os.path.exists(self.feature_file_path):
            raise FileNotFoundError(f'Could not find features file -> {self.feature_file_path}')
        logger.info(f'Loading features from -> {self.feature_file_path}')
        with open(self.feature_file_path, 'r', encoding='utf-8') as file:
            self.all_features = json.load(file)
        logger.trace(f'Features -> {json.dumps(self.all_features, indent=4)}')

    def get_rank_frequency_based_features(self) -> list:
        """Get ranked features"""
        if self.all_features is None:
            self.load_features()

        store = {}
        for state_name in self.all_features.keys():
            for job_name in self.all_features[state_name].keys():
                rank_score = 1000
                for features in self.all_features[state_name][job_name]:
                    for feature in [features]:
                        if feature not in store:
                            store[feature] = rank_score
                        else:
                            store[feature] += rank_score
                        rank_score -= 1

        sorted_store = {k: v for k, v in sorted(store.items(), key=lambda item: item[1], reverse=True)}
        sorted_store = list(sorted_store.keys())
        return_top = self.n_top_features
        top_features = sorted_store[:return_top]
        logger.info(
            f'Rank frequency based top {min(return_top, len(top_features))} features -> {json.dumps(top_features, indent=4)}'
        )
        return top_features

    def summarise_selection(self, all_features) -> None:
        """Summarise selection results over all seeds"""
        for job_name in self.job_names:
            out_dir = os.path.join(self.output_dir, job_name)
            job_scores = {}
            for seed in self.seeds:
                features = all_features[str(seed)][job_name]
                scores = len(features) * [1]
                scores[: min(10, len(features))] = range(
                    10, 10 - min(10, len(features)), -1
                )  # first min(10, len(features)) features get rank score, rest get score of 1
                for i, feature in enumerate(features):
                    if feature in job_scores.keys():
                        job_scores[feature] += scores[i]
                    else:
                        job_scores[feature] = scores[i]

            job_scores = pd.DataFrame(job_scores.items(), columns=['feature', 'score'])
            job_scores = job_scores.sort_values(by='score', ascending=True).reset_index(drop=True)
            job_scores['score'] = job_scores['score'] / job_scores['score'].sum()

            ax = job_scores.plot.barh(x='feature', y='score', figsize=(10, 10))
            fig = ax.get_figure()
            plt.title(f'Feature importance over all random seeds')
            plt.xlabel('Feature importance')
            plt.tight_layout()
            plt.gca().legend_.remove()
            plt.savefig(os.path.join(out_dir, f'feature_importance.pdf'), dpi=fig.dpi)
            plt.close(fig)

            job_scores = job_scores.iloc[-self.n_top_features:, :]
            ax = job_scores.plot.barh(x='feature', y='score')
            fig = ax.get_figure()
            plt.title(f'Feature importance over all random seeds (top {self.n_top_features})')
            plt.xlabel('Feature importance')
            plt.tight_layout()
            plt.gca().legend_.remove()
            plt.savefig(os.path.join(out_dir, f'feature_importance_top{self.n_top_features}.pdf'), dpi=fig.dpi)
            plt.close(fig)

    def summarise_verification(self) -> None:
        """Summarise verification results over all seeds"""
        v_scoring_dict = self.config.verification.scoring[self.config.meta.learn_task]
        self.verif_scoring = [v_scoring for v_scoring in v_scoring_dict if v_scoring_dict[v_scoring]]

        fig_roc_jobs, ax_roc_jobs = plt.subplots()
        fig_prc_jobs, ax_prc_jobs = plt.subplots()
        for job_name in self.job_names:
            out_dir = os.path.join(self.output_dir, job_name)
            fig_roc_models, ax_roc_models = plt.subplots()
            fig_prc_models, ax_prc_models = plt.subplots()
            for model in self.models + self.ensemble:
                fig_roc_baseline, ax_roc_baseline = plt.subplots()
                fig_prc_baseline, ax_prc_baseline = plt.subplots()

                self.average_scores(job_name, model)
                self.plot_auc(ax_roc_models, ax_prc_models, label=f'{model}')
                self.plot_auc(ax_roc_baseline, ax_prc_baseline, label=f'Top {self.n_top_features} features')
                self.average_scores('all_features', model)
                self.plot_auc(ax_roc_baseline, ax_prc_baseline, label=f'All features')
                self.save_plots(
                    fig_roc_baseline,
                    ax_roc_baseline,
                    fig_prc_baseline,
                    ax_prc_baseline,
                    self.pos_rate,
                    os.path.join(out_dir, model),
                    'baseline.pdf',
                )

            self.save_plots(
                fig_roc_models,
                ax_roc_models,
                fig_prc_models,
                ax_prc_models,
                self.pos_rate,
                out_dir,
                'all_models.pdf',
            )

    def average_scores(self, job_name, model) -> None:
        tprs = []
        precisions = []
        self.mean_x = np.linspace(0, 1, 100)
        self.averaged_scores = {score: [] for score in self.verif_scoring}
        for seed in self.seeds:
            scores = self.get_store('score', str(seed), job_name)[model]
            for score in self.verif_scoring:
                self.averaged_scores[score].append(scores[score])
            interp_tpr = np.interp(self.mean_x, scores['fpr'], scores['tpr'])  # AUROC
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            interp_precision = np.interp(self.mean_x, scores['precision'], scores['recall'])  # AUPRC
            interp_precision[0] = 1.0
            precisions.append(interp_precision)

        self.averaged_scores = {
            score: f'{np.mean(self.averaged_scores[score]):.3f} +- {np.std(self.averaged_scores[score]):.3f}'
            for score in self.verif_scoring
        }  # compute mean +- std for all scores
        self.mean_tpr = np.mean(tprs, axis=0)  # compute mean +- std for AUC plots
        self.std_tpr = np.std(tprs, axis=0)
        self.mean_precision = np.mean(precisions, axis=0)  # AUPRC
        self.std_precision = np.std(precisions, axis=0)
        self.pos_rate = scores['pos_rate']

        self.tprs_upper = np.minimum(self.mean_tpr + self.std_tpr, 1)
        self.tprs_lower = np.maximum(self.mean_tpr - self.std_tpr, 0)
        self.precisions_upper = np.minimum(self.mean_precision + self.std_precision, 1)
        self.precisions_lower = np.maximum(self.mean_precision - self.std_precision, 0)

    def plot_auc(self, ax_roc, ax_prc, label) -> None:
        ax_roc.plot(
            self.mean_x, self.mean_tpr, label=f'{label}, AUROC={self.averaged_scores["roc_auc_score"]}', alpha=0.7
        )
        ax_roc.fill_between(self.mean_x, self.tprs_lower, self.tprs_upper, alpha=0.1)
        ax_prc.plot(
            self.mean_x,
            self.mean_precision,
            label=f'{label}, AUPRC={self.averaged_scores["average_precision_score"]}',
            alpha=0.7,
        )
        ax_prc.fill_between(self.mean_x, self.precisions_lower, self.precisions_upper, alpha=0.1)

    def save_plots(self, fig_roc, ax_roc, fig_prc, ax_prc, pos_rate, out_dir, name: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        ax_roc.set_title('Receiver-operator curve (ROC)')
        ax_roc.set_xlabel('1 - Specificity')
        ax_roc.set_ylabel('Sensitivity')
        ax_roc.grid()
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Baseline, AUROC=0.5', alpha=0.7)  # baseline
        ax_roc.legend()
        fig_roc.savefig(os.path.join(out_dir, f'AUROC_{name}'))
        fig_roc.clear()
        ax_prc.set_title('Precision-recall curve (PRC)')
        ax_prc.set_xlabel('Recall (Sensitivity)')
        ax_prc.set_ylabel('Precision')
        ax_prc.grid()
        ax_prc.axhline(
            y=pos_rate, color='k', linestyle='--', label=f'Baseline, AUPRC={pos_rate}', alpha=0.7
        )  # baseline
        ax_prc.legend()
        fig_prc.savefig(os.path.join(out_dir, f'AUPRC_{name}'))
        fig_prc.clear()
