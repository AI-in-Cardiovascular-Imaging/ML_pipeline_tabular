import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from roc_utils import compute_roc, plot_mean_roc

from feature_corr.crates.helpers import job_name_cleaner
from feature_corr.data_borg import DataBorg, NestedDefaultDict


class Report(DataBorg):
    """Class to summarise and report all results"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        experiment_name = config.meta.name
        self.seeds = config.meta.seed
        self.output_dir = os.path.join(config.meta.output_dir, experiment_name)
        OmegaConf.save(
            config, os.path.join(self.output_dir, 'job_config.yaml')
        )  # save copy of config for future reference
        self.plot_format = config.meta.plot_format
        self.n_bootstraps = config.data_split.n_bootstraps
        self.jobs = config.selection.jobs
        self.job_names = job_name_cleaner(self.jobs)
        self.n_top_features = config.verification.use_n_top_features
        if isinstance(self.n_top_features, str):  # turn range provided as string into list
            self.n_top_features = list(eval(self.n_top_features))
            config.verification.use_n_top_features = self.n_top_features
        models_dict = config.verification.models
        self.models = [model for model in models_dict if models_dict[model]]
        self.learn_task = config.meta.learn_task
        self.learn_task = 'binary_classification'
        self.opt_scoring = config.selection.scoring[self.learn_task]
        scoring_dict = config.verification.scoring[self.learn_task]
        self.rep_scoring = [v_scoring for v_scoring in scoring_dict if scoring_dict[v_scoring]]
        self.rep_scoring.append('pos_rate')
        if config.meta.overwrite:
            self.load_intermediate_results(self.output_dir)
        else:
            for seed in self.seeds:  # initialise empty score containers to be filled during verification
                for job_name in self.job_names:
                    for n_top in self.n_top_features:
                        scores = NestedDefaultDict()
                        for model in self.models:
                            scores[model] = {score: [] for score in self.rep_scoring + ['true', 'pred']}
                        self.set_store('score', str(seed), f'{job_name}_{n_top}', scores)
                scores = NestedDefaultDict()
                for model in self.models:
                    scores[model] = {score: [] for score in self.rep_scoring + ['true', 'pred']}
                self.set_store('score', str(seed), 'all_features', scores)

        self.ensemble = [model for model in self.models if 'ensemble' in model]  # only ensemble models
        self.models = [model for model in self.models if model not in self.ensemble]
        if len(self.models) < 2:  # ensemble methods need at least two models two combine their results
            self.ensemble = []
        self.all_features = None

    def __call__(self):
        """Run feature report"""
        self.summarise_selection()
        self.summarise_verification()

    def summarise_selection(self) -> None:
        """Summarise selection results over all seeds"""
        for job_name in self.job_names:
            out_dir = os.path.join(self.output_dir, job_name)
            job_scores = self.get_store('feature_score', None, job_name)
            job_scores = pd.DataFrame(job_scores.items(), columns=['feature', 'score'])
            job_scores = job_scores.sort_values(by='score', ascending=True).reset_index(drop=True)
            job_scores['score'] = job_scores['score'] / job_scores['score'].sum()

            ax = job_scores.plot.barh(x='feature', y='score', figsize=(10, 10))
            fig = ax.get_figure()
            plt.title(f'Average feature importance')
            plt.xlabel('Average feature importance')
            plt.tight_layout()
            plt.gca().legend_.remove()
            plt.savefig(os.path.join(out_dir, f'avg_feature_importance_all.{self.plot_format}'), dpi=fig.dpi)
            plt.close(fig)

            for n_top in range(5, max(self.n_top_features), 10):
                job_scores_n_top = job_scores.iloc[-n_top:, :]
                ax = job_scores_n_top.plot.barh(x='feature', y='score')
                fig = ax.get_figure()
                plt.title(f'Average feature importance (top {n_top})')
                plt.xlabel('Average feature importance')
                plt.tight_layout()
                plt.gca().legend_.remove()
                plt.savefig(
                    os.path.join(out_dir, f'avg_feature_importance_top{n_top}.{self.plot_format}'),
                    dpi=fig.dpi,
                )
                plt.close(fig)

    def summarise_verification(self) -> None:
        """Summarise verification results over all seeds and bootstraps"""
        for job_name in self.job_names:
            out_dir = os.path.join(self.output_dir, job_name)
            with open(os.path.join(out_dir, f'results_{self.n_bootstraps}_bootstraps.txt'), 'w') as file:
                for model in self.models + self.ensemble:
                    best_opt_score, higher_is_better = self.init_scoring()
                    best_auroc = None
                    file.write(f'Results for {model} model:\n' 'All features:\n')
                    _, _, avg_scores, _ = self.average_scores('all_features', model)
                    [file.write(f'\t{k}: {v}\n') for k, v in avg_scores.items()]
                    mean = []
                    std = []
                    for n_top in self.n_top_features:  # compute average scores and populate plots
                        file.write(f'Top {n_top} features:\n')
                        mean_scores, std_scores, avg_scores, aurocs = self.average_scores(f'{job_name}_{n_top}', model)
                        mean_opt_score = mean_scores[f'{self.opt_scoring}_score']
                        if (higher_is_better and mean_opt_score > best_opt_score) or (
                            not higher_is_better and mean_opt_score < best_opt_score
                        ):
                            best_opt_score = mean_opt_score
                            best_auroc = aurocs
                        mean.append(mean_opt_score)
                        std.append(std_scores[f'{self.opt_scoring}_score'])
                        [file.write(f'\t{k}: {v}\n') for k, v in avg_scores.items()]

                    plt.figure()
                    plt.xlabel('Number of features selected')
                    plt.ylabel(f'Mean {self.opt_scoring}')
                    plt.grid(alpha=0.5)
                    plt.errorbar(
                        self.n_top_features,
                        mean,
                        yerr=std,
                    )
                    plt.title(f'{model} model performance for increasing number of features')
                    plt.savefig(
                        os.path.join(out_dir, f'results_{model}_{self.n_bootstraps}_bootstraps.{self.plot_format}')
                    )
                    plt.clf()
                    
                    plt.figure()
                    plot_mean_roc(best_auroc, show_ci=True, show_ti=False)
                    plt.title(f'Mean ROC for {model} model')
                    plt.savefig(
                        os.path.join(out_dir, f'AUROC_{model}_{self.n_bootstraps}_bootstraps.{self.plot_format}')
                    )
                    plt.clf()

    def average_scores(self, job_name, model) -> None:
        """Average results over all seeds and bootstraps"""
        averaged_scores = {score: [] for score in self.rep_scoring}
        aurocs = []
        for seed in self.seeds:
            scores = self.get_store('score', str(seed), job_name)[model]  # contains results for all bootstraps for seed
            for score in self.rep_scoring:
                averaged_scores[score].append(scores[score])
            for boot_iter in range(self.n_bootstraps):
                aurocs.append(compute_roc(scores['pred'][boot_iter], scores['true'][boot_iter], pos_label=True))
        mean_scores = {score: np.mean(averaged_scores[score]) for score in self.rep_scoring}
        std_scores = {score: np.std(averaged_scores[score]) for score in self.rep_scoring}
        averaged_scores = {
            score: f'{mean_scores[score]:.3f} +- {std_scores[score]:.3f}' for score in self.rep_scoring
        }  # compute mean +- std for all scores
        return mean_scores, std_scores, averaged_scores, aurocs

    def init_scoring(self):
        """Find value corresponding to a bad score given the scoring metric, and return whether higher is better"""
        if self.opt_scoring in ['roc_auc', 'average_precision', 'precision', 'recall', 'f1', 'accuracy', 'r2']:
            return 0, True
        elif self.opt_scoring in ['mean_absolute_error', 'mean_squared_error']:
            return np.Inf, False
        else:
            raise NotImplementedError
