import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import sklearn.metrics as metrics
import imblearn.metrics as imb_metrics
from loguru import logger
from roc_utils import compute_roc, plot_mean_roc

from pipeline_tabular.config_manager import ConfigManager
from pipeline_tabular.utils.helpers import generate_seeds, job_name_cleaner
from pipeline_tabular.data_handler.data_handler import DataHandler
from pipeline_tabular.utils.explain.explain import Explain  # import here to avoid circular imports


class CollectResults(DataHandler):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_dir = config.meta.output_dir
        self.plot_format = config.meta.plot_format
        self.learn_task = config.meta.learn_task
        self.init_seed = config.data_split.init_seed
        self.n_seeds = config.data_split.n_seeds
        self.n_bootstraps = config.data_split.n_bootstraps
        np.random.seed(self.init_seed)
        self.seeds = generate_seeds(self.init_seed, self.n_seeds)
        self.opt_scoring = config.selection.scoring[self.learn_task]
        self.jobs = config.selection.jobs
        self.job_names = job_name_cleaner(self.jobs)
        self.n_top_features = config.verification.use_n_top_features
        models_dict = config.verification.models
        self.rep_models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [model for model in self.rep_models if 'ensemble' in model]  # only ensemble models
        self.rep_models = [model for model in self.rep_models if model not in self.ensemble]
        if len(self.rep_models) < 2:  # ensemble methods need at least two models to combine their results
            self.ensemble = []
        self.to_collect = config.collect_results.experiments
        metrics_dict = config.collect_results.metrics_to_collect[self.learn_task]
        self.metrics_to_collect = [metric for metric in metrics_dict if metrics_dict[metric]]
        if f'{self.opt_scoring}_score' not in self.metrics_to_collect:  # ensure optimisation metric is always collected
            self.metrics_to_collect.append(f'{self.opt_scoring}_score')
        self.metrics_to_plot = [metric for metric in self.metrics_to_collect if metric != 'roc']
        self.all_features = None

    def __call__(self) -> None:
        self.explainer = Explain(self.config)
        self.collect_results()

    def collect_results(self):
        """Collect results over all experiments, jobs, models, seeds and bootstraps and summarise them"""
        self.results = pd.DataFrame(
            columns=['experiment', 'best_job', 'best_model', 'best_n_top'] + self.metrics_to_plot,
            index=range(len(self.to_collect)),
        )
        for experiment_index, experiment_name in enumerate(self.to_collect):
            logger.info(f'Collecting results for experiment {experiment_name}...')
            self.report_dir = os.path.join(self.out_dir, experiment_name, 'report')
            os.makedirs(self.report_dir, exist_ok=True)
            self.load_intermediate_results(os.path.join(self.out_dir, experiment_name))
            self.summarise_selection(experiment_name)
            self.summarise_verification(experiment_index, experiment_name)

        self.summarise_experiments()

        for metric in self.metrics_to_plot:
            self.results[metric] = self.results[metric].apply(np.mean)
        self.results.to_csv(os.path.join(self.out_dir, 'results.csv'), index=False)

    def summarise_selection(self, experiment_name) -> None:
        """Summarise selection results over all seeds"""
        for job_name in self.job_names:
            out_dir = os.path.join(self.out_dir, experiment_name, job_name)
            job_scores = self.get_store('feature_score', None, job_name)
            job_scores = pd.DataFrame(job_scores.items(), columns=['feature', 'score'])
            job_scores = job_scores.sort_values(by='score', ascending=True).reset_index(drop=True)
            job_scores['score'] = job_scores['score'] / job_scores['score'].sum()

            try:
                ax = job_scores.plot.barh(x='feature', y='score', figsize=(10, 10))
            except TypeError:  # no data is available to plot, i.e. collect_results flag set to True by accident
                logger.error(f'No results found to collect for job {job_name}.')
                raise SystemExit(0)
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

    def summarise_verification(self, experiment_index, experiment_name) -> None:
        """Summarise verification results over all seeds and bootstraps"""
        verification_scores = {}
        for metric in self.metrics_to_collect + ['n_top']:
            verification_scores[metric] = pd.DataFrame(columns=self.job_names, index=(self.rep_models + self.ensemble))
        verification_scores = self.average_scores(verification_scores)
        mean_verification_scores = self.reduce_scores(verification_scores, np.mean)
        mean_opt_scores = mean_verification_scores[f'{self.opt_scoring}_score']
        best_models = {job_name: mean_opt_scores[job_name].idxmax() for job_name in self.job_names}
        # self.explainer(
        #     experiment_name,
        #     verification_scores,
        #     self.opt_scoring,
        #     self.job_names,
        #     best_models,
        #     self.seeds,
        #     self.n_bootstraps,
        # )

        if 'roc' in self.metrics_to_collect:
            self.plot_rocs(verification_scores['roc'], best_models)
        self.plot_heatmaps(mean_verification_scores)
        logger.info(
            f'\nStrategies summary:\n' + '\n'.join(f'Strat. {i+1}: {job}' for i, job in enumerate(self.job_names))
        )

        # fill results dataframe
        best_model_index, best_job_index = [
            x[0] for x in np.unravel_index([np.argmax(mean_opt_scores.values)], mean_opt_scores.values.shape)
        ]
        clean_experiment_name = experiment_name.split('_')[-1]
        clean_job_name = f'Strat. {best_job_index+1}'
        best_model, best_job = mean_opt_scores.index[best_model_index], mean_opt_scores.columns[best_job_index]
        self.results.loc[experiment_index] = [clean_experiment_name, clean_job_name, best_model] + [
            verification_scores[metric].loc[best_model][best_job] for metric in ['n_top'] + self.metrics_to_plot
        ]

    def summarise_experiments(self) -> None:
        """Summarise results across experiments"""
        self.results = self.results.explode(self.metrics_to_plot)  # expand lists into columns
        for metric in self.metrics_to_plot:
            fig = plt.figure()
            sns.boxplot(data=self.results, x='experiment', y=metric)
            plt.tight_layout()
            fig.savefig(os.path.join(self.out_dir, f'{metric}_boxplot.{self.plot_format}'))
            plt.close(fig)

    def average_scores(self, verification_scores) -> None:
        """Average results over all seeds and bootstraps"""
        for job_name in self.job_names:
            for model in self.rep_models + self.ensemble:
                best_mean_opt_score, higher_is_better = self.init_scoring()
                best_all_scores = None
                best_roc = None
                best_n_top = None
                for n_top in self.n_top_features:  # find best number of features for each job/model combination
                    all_scores, roc = self.collect_scores(f'{job_name}_{n_top}', model)
                    mean_opt_score = np.mean(all_scores[f'{self.opt_scoring}_score'])
                    if higher_is_better and mean_opt_score > best_mean_opt_score:
                        best_mean_opt_score = mean_opt_score
                        best_all_scores = all_scores
                        best_roc = roc
                        best_n_top = n_top
                for metric in self.metrics_to_collect:
                    if metric == 'roc':
                        continue
                    best_all_scores[metric] = [elem for sub in best_all_scores[metric] for elem in sub]
                    verification_scores[metric].loc[model, job_name] = best_all_scores[metric]
                verification_scores['roc'].loc[model, job_name] = best_roc
                verification_scores['n_top'].loc[model, job_name] = best_n_top

        return verification_scores

    def collect_scores(self, job_name, model) -> None:
        """Collect results over all seeds and bootstraps"""
        all_scores = {score: [] for score in self.metrics_to_collect}
        roc = []
        for seed_index, seed in enumerate(self.seeds):
            try:
                scores = self.get_store('score', seed, job_name)[model]
            except KeyError:  # model not yet stored for this seed/job
                scores = {scoring: [] for scoring in self.metrics_to_collect}

            if scores[list(scores.keys())[0]]:  # else scores empty, i.e. not run for this job_name/n_top
                for score in self.metrics_to_collect:
                    if score == 'roc':
                        for boot_iter in range(self.n_bootstraps):
                            roc.append(
                                compute_roc(scores['probas'][boot_iter], scores['true'][boot_iter], pos_label=True)
                            )
                        continue
                    if score not in scores.keys() or len(scores[score]) < self.n_bootstraps:  # score not yet computed
                        scores = self.compute_missing_scores(scores, score)
                    all_scores[score].append(scores[score])
            else:
                np.delete(self.seeds, seed_index)

        return all_scores, roc

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

        return scores

    def reduce_scores(self, verification_scores, function):
        reduced_verification_scores = {
            metric: pd.DataFrame(
                np.vectorize(function)(verification_scores[metric]),
                index=verification_scores[metric].index,
                columns=verification_scores[metric].columns,
            )
            for metric in self.metrics_to_plot  # cannot reduce roc
        }
        return reduced_verification_scores

    def plot_rocs(self, rocs, best_models):
        roc_plot, roc_ax = plt.subplots()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for job_index, job_name in enumerate(self.job_names):
            best_roc_job = rocs.loc[best_models[job_name], job_name]
            if best_roc_job is not None:
                plot_mean_roc(
                    best_roc_job,
                    show_ci=False,
                    show_ti=False,
                    show_opt=False,
                    ax=roc_ax,
                    label=f'Strat. {job_index+1}',
                    color=colors[job_index],
                )
                plt.figure()
                plot_mean_roc(
                    best_roc_job,
                    show_ci=True,
                    show_ti=False,
                    show_opt=False,
                    label=f'Strat. {job_index+1}',
                )
                plt.title(f'Best mean ROC for Strat. {job_index+1}')
                plt.savefig(os.path.join(self.report_dir, f'AUROC_best_strat_{job_index+1}.{self.plot_format}'))
                plt.clf()

        roc_ax.set_title('Best mean ROC for all strategies')
        roc_plot.savefig(os.path.join(self.report_dir, f'AUROC_best_per_strat.{self.plot_format}'))

    def plot_heatmaps(self, verification_scores):
        cmaps = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys', 'YlGnBu', 'YlOrRd', 'PuBu', 'PuRd']
        for i, score in enumerate(self.metrics_to_plot):
            fig = plt.figure()
            sns.heatmap(
                verification_scores[score],
                annot=True,
                xticklabels=[f'Strat. {i+1}' for i in range(len(self.job_names))],
                yticklabels=True,
                vmin=0.0,
                vmax=1.0,
                cmap=cmaps[i],
                fmt='.2g',
            )
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_dir, f'results_heatmap_{self.metrics_to_plot[i]}.{self.plot_format}'))
            plt.close(fig)

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


if __name__ == '__main__':
    config = ConfigManager()()
    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)
    if config.meta.ignore_warnings:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    CollectResults(config)()
