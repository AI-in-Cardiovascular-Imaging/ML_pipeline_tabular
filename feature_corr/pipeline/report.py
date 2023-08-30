import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import sklearn.metrics as metrics
import imblearn.metrics as imb_metrics
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from roc_utils import compute_roc, plot_mean_roc

from feature_corr.utils.helpers import job_name_cleaner
from feature_corr.data_handler.data_handler import DataHandler, NestedDefaultDict


class Report(DataHandler):
    """Class to summarise and report all results"""

    def __init__(self, config: DictConfig, seeds) -> None:
        super().__init__()
        self.config = config
        self.seeds = seeds
        self.output_dir = os.path.join(config.meta.output_dir, config.meta.name)
        self.plot_format = config.meta.plot_format
        self.learn_task = config.meta.learn_task
        self.n_bootstraps = config.data_split.n_bootstraps
        self.n_top_features = config.verification.use_n_top_features
        if isinstance(self.n_top_features, str):  # turn range provided as string into list
            self.n_top_features = list(eval(self.n_top_features))
            config.verification.use_n_top_features = self.n_top_features
        models_dict = config.verification.models
        self.models = [model for model in models_dict if models_dict[model]]
        self.opt_scoring = config.selection.scoring[self.learn_task]
        self.jobs = config.selection.jobs
        self.job_names = job_name_cleaner(self.jobs)
        scoring_dict = config.verification.scoring[self.learn_task]
        self.rep_scoring = [v_scoring for v_scoring in scoring_dict if scoring_dict[v_scoring]]
        self.rep_scoring.append('pos_rate')
        self.ensemble = [model for model in self.models if 'ensemble' in model]  # only ensemble models
        self.models = [model for model in self.models if model not in self.ensemble]
        if len(self.models) < 2:  # ensemble methods need at least two models two combine their results
            self.ensemble = []
        self.init_containers()
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
            plt.title(f'Average feature ranking')
            plt.xlabel('Average feature ranking')
            plt.tight_layout()
            plt.gca().legend_.remove()
            plt.savefig(os.path.join(out_dir, f'avg_feature_ranking_all.{self.plot_format}'), dpi=fig.dpi)
            plt.close(fig)

            for n_top in range(5, max(self.n_top_features), 10):
                job_scores_n_top = job_scores.iloc[-n_top:, :]
                ax = job_scores_n_top.plot.barh(x='feature', y='score')
                fig = ax.get_figure()
                plt.title(f'Average feature ranking (top {n_top})')
                plt.xlabel('Average feature ranking')
                plt.tight_layout()
                plt.gca().legend_.remove()
                plt.savefig(
                    os.path.join(out_dir, f'avg_feature_ranking_top{n_top}.{self.plot_format}'),
                    dpi=fig.dpi,
                )
                plt.close(fig)

    def summarise_verification(self) -> None:
        """Summarise verification results over all seeds and bootstraps"""
        best_mean_opt_scores = pd.DataFrame(columns=self.job_names, index=(self.models + self.ensemble))
        best_all_scores = pd.DataFrame(
            columns=[f'Strat. {i+1}' for i in range(len(self.job_names))],
            index=(['job', 'model', '#features'] + self.rep_scoring)
            + ['MWU p-value'],
        )
        best_opt_scores = []
        roc_plot, roc_ax = plt.subplots()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i, job_name in enumerate(self.job_names):
            out_dir = os.path.join(self.output_dir, job_name)
            best_mean_opt_score_job, higher_is_better = self.init_scoring()
            best_mean_opt_scores_job = []
            best_roc_job = []
            best_opt_score = None
            best_model = None
            best_n_top = None
            best_scores_mean = None
            best_scores_std = None
            with open(os.path.join(out_dir, f'results.txt'), 'w') as file:
                for model in self.models + self.ensemble:
                    best_opt_score_model, _ = self.init_scoring()
                    best_roc_model = None
                    mean = []
                    std = []
                    for n_top in self.n_top_features:  # compute average scores and populate plots
                        file.write(f'Top {n_top} features:\n')
                        mean_scores, std_scores, opt_scores, avg_scores, roc = self.average_scores(
                            f'{job_name}_{n_top}', model
                        )
                        mean_opt_score = mean_scores[f'{self.opt_scoring}_score']
                        if (higher_is_better and mean_opt_score > best_opt_score_model) or (
                            not higher_is_better and mean_opt_score < best_opt_score_model
                        ):  # update best scores for model
                            best_opt_score_model = mean_opt_score
                            best_roc_model = roc
                            if (higher_is_better and mean_opt_score > best_mean_opt_score_job) or (
                                not higher_is_better and mean_opt_score < best_mean_opt_score_job
                            ):  # update best scores for job
                                best_mean_opt_score_job = mean_opt_score
                                best_opt_score = opt_scores
                                best_model = model
                                best_n_top = n_top
                                best_scores_mean = mean_scores
                                best_scores_std = std_scores
                        mean.append(mean_opt_score)
                        std.append(std_scores[f'{self.opt_scoring}_score'])
                        [file.write(f'\t{k}: {v}\n') for k, v in avg_scores.items()]

                    best_mean_opt_scores_job.append(best_opt_score_model)
                    best_roc_job.append(best_roc_model)
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
                    plt.savefig(os.path.join(out_dir, f'results_{model}.{self.plot_format}'))
                    plt.clf()

                    plt.figure()
                    plot_mean_roc(best_roc_model, show_ci=True, show_ti=False, show_opt=False)
                    plt.title(f'Mean ROC for {model} model')
                    plt.savefig(os.path.join(out_dir, f'AUROC_best_{model}.{self.plot_format}'))
                    plt.clf()

            best_opt_scores.append(best_opt_score)
            best_mean_opt_scores[job_name] = best_mean_opt_scores_job
            if i == 0:  # cannot compare job 0 to itself
                stats = ['-']
            else:
                stats = [
                    round(pg.mwu(best_opt_scores[0], best_opt_scores[i])['p-val'][0], 4),
                ]
            best_all_scores[f'Strat. {i+1}'] = (
                [job_name, best_model, best_n_top]
                + [
                    f'{mean:.2f} \u00B1 {std:.2f}'
                    for mean, std in zip(best_scores_mean.values(), best_scores_std.values())
                ]
                + stats
            )
            best_index = (
                np.argmax(best_mean_opt_scores_job) if higher_is_better else np.argmin(best_mean_opt_scores_job)
            )
            best_roc_job = best_roc_job[best_index]
            plot_mean_roc(
                best_roc_job,
                show_ci=False,
                show_ti=False,
                show_opt=False,
                ax=roc_ax,
                label=f'Strat. {i+1}',
                color=colors[i],
            )
            plt.figure()
            plot_mean_roc(
                best_roc_job,
                show_ci=True,
                show_ti=False,
                show_opt=False,
                label=f'Strat. {i+1}',
            )
            plt.title(f'Best mean ROC for Strat. {i+1}')
            plt.savefig(os.path.join(self.output_dir, f'AUROC_best_strat_{i+1}.{self.plot_format}'))
            plt.clf()

        best_all_scores.to_csv(os.path.join(self.output_dir, f'best_model_all_scores.csv'))

        roc_ax.set_title('Best mean ROC for all strategies')
        roc_plot.savefig(os.path.join(self.output_dir, f'AUROC_best_per_strat.{self.plot_format}'))
        fig = plt.figure()
        sns.heatmap(
            best_mean_opt_scores,
            annot=True,
            xticklabels=[f'Strat. {i+1}' for i in range(len(self.job_names))],
            yticklabels=True,
            vmin=0.5,
            vmax=1.0,
            cmap='Blues',
            fmt='.2g',
        )
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'results_heatmap.{self.plot_format}'))
        plt.close(fig)
        logger.info(
            f'\nStrategies summary:\n' + '\n'.join(f'Strat. {i+1}: {job}' for i, job in enumerate(self.job_names))
        )

    def average_scores(self, job_name, model) -> None:
        """Average results over all seeds and bootstraps"""
        all_scores = {score: [] for score in self.rep_scoring}
        roc = []
        for seed in self.seeds:
            try:
                scores = self.get_store('score', seed, job_name)[model]
            except KeyError:  # model not yet stored for this seed/job
                scores = {scoring: [] for scoring in self.verif_scoring}
                
            if scores[list(scores.keys())[0]]:  # else scores empty, i.e. not run for this job_name/n_top
                for score in self.rep_scoring:
                    if score not in scores.keys() or len(scores[score]) < self.n_bootstraps:  # score not yet computed
                        self.compute_missing_scores(scores, score)
                    all_scores[score].append(scores[score])
                for boot_iter in range(self.n_bootstraps):
                    roc.append(compute_roc(scores['probas'][boot_iter], scores['true'][boot_iter], pos_label=True))
        mean_scores = {score: np.mean(all_scores[score]) for score in self.rep_scoring}
        std_scores = {score: np.std(all_scores[score]) for score in self.rep_scoring}
        averaged_scores = {
            score: f'{mean_scores[score]:.2f} +- {std_scores[score]:.2f}' for score in self.rep_scoring
        }  # compute mean +- std for all scores
        opt_scores = all_scores[f'{self.opt_scoring}_score']
        opt_scores = [item for sublist in opt_scores for item in sublist]  # flatten list
        return mean_scores, std_scores, opt_scores, averaged_scores, roc

    def compute_missing_scores(self, scores, score):
        try:  # some metrics can be calculated using probabilities, others need prediction
            scores[score] = [
                getattr(metrics, score)(scores['true'][boot_iter], scores['probas'][boot_iter])
                for boot_iter in range(self.n_bootstraps)
            ]
        except ValueError:
            scores[score] = [
                getattr(metrics, score)(scores['true'][boot_iter], scores['pred'][boot_iter])
                for boot_iter in range(self.n_bootstraps)
            ]
        except AttributeError:  # try imbalanced learn metrics (e.g. for specificity)
            try:
                scores[score] = [
                    getattr(imb_metrics, score)(scores['true'][boot_iter], scores['probas'][boot_iter])
                    for boot_iter in range(self.n_bootstraps)
                ]
            except ValueError:
                scores[score] = [
                    getattr(imb_metrics, score)(scores['true'][boot_iter], scores['pred'][boot_iter])
                    for boot_iter in range(self.n_bootstraps)
                ]

    def init_containers(self):
        if not self.config.meta.overwrite:
            scores_found = self.load_intermediate_results(self.output_dir)  # try loading available results
        if self.config.meta.overwrite or not scores_found:
            OmegaConf.save(
                self.config, os.path.join(self.output_dir, 'job_config.yaml')
            )  # save copy of config for future reference
            for seed in self.seeds:  # initialise empty score containers to be filled during verification
                for job_name in self.job_names:
                    for n_top in self.n_top_features:
                        scores = NestedDefaultDict()
                        for model in self.models + self.ensemble:
                            scores[model] = {score: [] for score in self.rep_scoring + ['probas', 'true', 'pred']}
                        self.set_store('score', str(seed), f'{job_name}_{n_top}', scores)

    def init_scoring(self):
        """Find value corresponding to a bad score given the scoring metric, and return whether higher is better"""
        if self.opt_scoring in [
            'roc_auc',
            'average_precision',
            'precision',
            'recall',
            'specificity',
            'f1',
            'accuracy',
            'r2',
        ]:
            return -np.Inf, True
        elif self.opt_scoring in ['mean_absolute_error', 'mean_squared_error']:
            return np.Inf, False
        else:
            raise NotImplementedError
