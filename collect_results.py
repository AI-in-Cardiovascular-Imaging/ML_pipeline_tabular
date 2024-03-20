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
from omegaconf import OmegaConf
from roc_utils import plot_mean_roc
from pipeline_tabular.utils.roc_utils.roc_utils import compute_roc_aucopt

from pipeline_tabular.config_manager import ConfigManager
from pipeline_tabular.utils.helpers import generate_seeds, job_name_cleaner
from pipeline_tabular.data_handler.data_handler import DataHandler
from pipeline_tabular.utils.explain.explain import Explain  # import here to avoid circular imports


class CollectResults(DataHandler):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_dir = config.meta.output_dir
        self.results_dir = os.path.join(self.out_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.plot_format = config.meta.plot_format
        learn_task = config.meta.learn_task
        init_seed = config.data_split.init_seed
        np.random.seed(init_seed)
        n_seeds = config.data_split.n_seeds
        self.n_bootstraps = config.data_split.n_bootstraps
        self.seeds = generate_seeds(init_seed, n_seeds)
        self.opt_scoring = config.selection.scoring[learn_task]
        models_dict = config.verification.models
        self.rep_models = [model for model in models_dict if models_dict[model]]
        self.ensemble = [model for model in self.rep_models if 'ensemble' in model]  # only ensemble models
        self.rep_models = [model for model in self.rep_models if model not in self.ensemble]
        if len(self.rep_models) < 2:  # ensemble methods need at least two models to combine their results
            self.ensemble = []
        self.explain = config.collect_results.explain
        self.to_collect = config.collect_results.experiments
        metrics_dict = config.collect_results.metrics_to_collect[learn_task]
        self.use_youden_index = metrics_dict['youden_index']
        self.metrics_to_collect = [
            metric for metric in metrics_dict if metrics_dict[metric] and metric != 'youden_index'
        ]
        if f'{self.opt_scoring}_score' not in self.metrics_to_collect:  # ensure optimisation metric is always collected
            self.metrics_to_collect.append(f'{self.opt_scoring}_score')
        self.metrics_to_plot = [metric for metric in self.metrics_to_collect if metric not in ['roc']]

    def __call__(self) -> None:
        self.explainer = Explain(self.config)
        self.collect_results()

    def collect_results(self):
        """Collect results over all experiments, jobs, models, seeds and bootstraps and summarise them"""
        self.results = pd.DataFrame(
            columns=['experiment', 'best_job', 'best_model', 'best_n_top'] + self.metrics_to_plot,
            index=range(len(self.to_collect)),
        )
        self.opt_scores = {}
        self.best_jobs = {}
        self.best_models = {}
        self.best_models_per_job = {}
        self.clean_experiment_names = []
        for experiment_index, experiment_name in enumerate(self.to_collect):
            logger.info(f'Collecting results for experiment {experiment_name}...')
            experiment_dir = os.path.join(self.out_dir, experiment_name)
            self.report_dir = os.path.join(experiment_dir, 'report')
            os.makedirs(self.report_dir, exist_ok=True)
            try:
                experiment_config = OmegaConf.load(os.path.join(experiment_dir, 'job_config.yaml'))
            except FileNotFoundError:
                logger.warning(f'No job_config.yaml found in {experiment_dir}. Skipping experiment...')
                continue
            self.job_names = job_name_cleaner(experiment_config.selection.jobs)
            self.n_top_features = experiment_config.verification.use_n_top_features
            self.load_intermediate_results(experiment_dir)
            self.summarise_selection(experiment_name)
            self.summarise_verification(experiment_index, experiment_name)

        self.summarise_experiments()

    def summarise_selection(self, experiment_name) -> None:
        """Summarise selection results over all seeds"""
        for job_name in self.job_names:
            out_dir = os.path.join(self.out_dir, experiment_name, job_name)
            os.makedirs(out_dir, exist_ok=True)
            job_scores = self.get_store('feature_score', None, job_name)
            job_scores = pd.DataFrame(job_scores.items(), columns=['feature', 'score'])
            job_scores = job_scores.sort_values(by='score', ascending=True).reset_index(drop=True)
            job_scores['score'] = job_scores['score'] / job_scores['score'].sum()

            try:
                ax = job_scores.plot.barh(x='feature', y='score', figsize=(10, 10), color='green')
            except TypeError:  # no data is available to plot, i.e. collect_results flag set to True by accident
                logger.error(f'No results found to collect for job {job_name}.')
                raise SystemExit(0)
            fig = ax.get_figure()
            plt.title(f'Average feature ranking')
            plt.xlabel('Average feature ranking')
            plt.tight_layout()
            plt.gca().legend_.remove()
            plt.savefig(os.path.join(out_dir, f'avg_feature_ranking_all.{self.plot_format}'), dpi=300)
            plt.close(fig)

            for n_top in range(5, max(self.n_top_features), 10):
                job_scores_n_top = job_scores.iloc[-n_top:, :]
                ax = job_scores_n_top.plot.barh(x='feature', y='score', color='green')
                fig = ax.get_figure()
                plt.title(f'Average feature ranking (top {n_top})')
                plt.xlabel('Average feature ranking')
                plt.tight_layout()
                plt.gca().legend_.remove()
                plt.savefig(
                    os.path.join(out_dir, f'avg_feature_ranking_top{n_top}.{self.plot_format}'),
                    dpi=300,
                )
                plt.close(fig)

    def summarise_verification(self, experiment_index, experiment_name) -> None:
        """Summarise verification results over all seeds and bootstraps"""
        verification_scores = {}
        for metric in self.metrics_to_collect + ['n_top']:
            verification_scores[metric] = pd.DataFrame(columns=self.job_names, index=(self.rep_models + self.ensemble))
        verification_scores = self.average_scores(verification_scores)
        self.opt_scores[experiment_name] = verification_scores[
            f'{self.opt_scoring}_score'
        ]  # save optimisation scores for statistical tests
        mean_verification_scores = self.reduce_scores(verification_scores, np.mean)
        mean_opt_scores = mean_verification_scores[f'{self.opt_scoring}_score']
        best_models = {job_name: mean_opt_scores[job_name].idxmax() for job_name in self.job_names}
        if self.explain:
            self.explainer(
                experiment_name,
                verification_scores,
                self.opt_scoring,
                self.job_names,
                best_models,
                self.seeds,
                self.n_bootstraps,
            )
        if 'roc' in self.metrics_to_collect:
            self.plot_rocs(verification_scores['roc'], best_models)
        self.plot_heatmaps(mean_verification_scores)
        logger.info(
            f'\nStrategies summary:\n' + '\n'.join(f'Strat. {i+1}: {job}' for i, job in enumerate(self.job_names))
        )

        # fill results dataframe
        to_find = np.argmax if self.higher_is_better else np.argmin
        best_model_index, best_job_index = [
            x[0] for x in np.unravel_index([to_find(mean_opt_scores.values)], mean_opt_scores.values.shape)
        ]
        self.clean_experiment_names.append('_'.join(experiment_name.split('_')[2:]))
        clean_job_name = f'Strat. {best_job_index+1}'
        best_model, best_job = mean_opt_scores.index[best_model_index], mean_opt_scores.columns[best_job_index]
        self.best_jobs[experiment_name] = best_job
        self.best_models[experiment_name] = best_model
        self.best_models_per_job[experiment_name] = best_models
        self.results.loc[experiment_index] = [
            self.clean_experiment_names[experiment_index],
            clean_job_name,
            best_model,
        ] + [verification_scores[metric].loc[best_model][best_job] for metric in ['n_top'] + self.metrics_to_plot]

    def summarise_experiments(self) -> None:
        """Summarise results across experiments"""
        self.compute_statistics()
        self.save_mean_results()
        results_to_plot = self.results.explode(self.metrics_to_plot)  # expand lists into columns
        for metric in self.metrics_to_plot:
            fig = plt.figure()
            sns.boxplot(data=results_to_plot, x='experiment', y=metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fig.savefig(os.path.join(self.results_dir, f'{metric}_boxplot.{self.plot_format}'), dpi=300)
            plt.close(fig)

    def compute_statistics(self):
        best_aurocs = {}
        for experiment_name in self.to_collect:
            best_aurocs[experiment_name] = self.opt_scores[experiment_name].loc[self.best_models[experiment_name]][
                self.best_jobs[experiment_name]
            ]
            best_aurocs_all_jobs = self.opt_scores[experiment_name]
            exp_jobs = best_aurocs_all_jobs.columns
            clean_job_names = [f'Strat. {i+1}' for i in range(len(exp_jobs))]
            stats_all_jobs = pd.DataFrame(index=clean_job_names, columns=clean_job_names)
            for i, job_1 in enumerate(exp_jobs):
                for j, job_2 in enumerate(exp_jobs):
                    if i >= j:  # only compute upper triangle
                        continue
                    stats_all_jobs.iloc[i, j] = pg.mwu(
                        best_aurocs_all_jobs.loc[self.best_models_per_job[experiment_name][job_1]][job_1],
                        best_aurocs_all_jobs.loc[self.best_models_per_job[experiment_name][job_2]][job_2],
                    ).round(2)['p-val'][0]
            fig = plt.figure()
            # stats_all_jobs = stats_all_jobs.dropna(axis=1, how='all').dropna(axis=0, how='all')
            if not stats_all_jobs.empty:
                sns.heatmap(stats_all_jobs.astype(float), annot=True, cmap='Purples', fmt='.2f')
                plt.title('Mann-Whitney U test p-values for best AUROC')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                fig.savefig(os.path.join(self.results_dir, f'mwu_pvals_{experiment_name}.{self.plot_format}'), dpi=300)
                plt.close(fig)

        stats_all_exp = pd.DataFrame(index=self.to_collect, columns=self.to_collect)
        for i, exp_1 in enumerate(self.to_collect):
            for j, exp_2 in enumerate(self.to_collect):
                if i >= j:  # only compute upper triangle
                    continue
                stats_all_exp.loc[exp_1, exp_2] = pg.mwu(best_aurocs[exp_1], best_aurocs[exp_2]).round(2)['p-val'][0]
        fig = plt.figure(figsize=(10, 10))
        # stats_all_exp = stats_all_exp.dropna(axis=1, how='all').dropna(axis=0, how='all')
        sns.heatmap(stats_all_exp.astype(float), annot=True, cmap='Purples', fmt='.2f')
        plt.title('Mann-Whitney U test p-values for best AUROC')
        plt.xticks(ticks=plt.xticks()[0], labels=self.clean_experiment_names, rotation=45, ha='right')
        plt.yticks(ticks=plt.yticks()[0], labels=self.clean_experiment_names, rotation=0)
        plt.tight_layout()
        fig.savefig(os.path.join(self.results_dir, f'mwu_pvals.{self.plot_format}'), dpi=300)
        plt.close(fig)

    def save_mean_results(self):
        mean_results = pd.DataFrame(index=self.results.index, columns=self.results.columns)
        info_cols = [col for col in mean_results.columns if col not in self.metrics_to_plot]
        mean_results[info_cols] = self.results[info_cols]
        mean_results[self.metrics_to_plot] = np.vectorize(np.mean)(self.results[self.metrics_to_plot])
        std_cols = [f'{col}_std' for col in self.metrics_to_plot]
        mean_results[std_cols] = np.vectorize(np.std)(self.results[self.metrics_to_plot])
        mean_results.to_csv(os.path.join(self.results_dir, 'mean_results.csv'), index=False, float_format='%.2f')

    def average_scores(self, verification_scores) -> None:
        """Average results over all seeds and bootstraps"""
        for job_index, job_name in enumerate(self.job_names):
            for model in self.rep_models + self.ensemble:
                best_mean_opt_score, self.higher_is_better = self.init_scoring()
                for n_top in self.n_top_features:  # find best number of features for each job/model combination
                    all_scores, roc, conf_matrix = self.collect_scores(f'{job_name}_{n_top}', model)
                    mean_opt_score = np.mean(all_scores[f'{self.opt_scoring}_score'])
                    if self.higher_is_better and mean_opt_score > best_mean_opt_score:
                        best_mean_opt_score = mean_opt_score
                        best_all_scores = all_scores
                        best_roc = roc
                        best_conf_matrix = conf_matrix
                        best_n_top = n_top
                for metric in self.metrics_to_collect:
                    if metric == 'roc':
                        continue
                    best_all_scores[metric] = [elem for sub in best_all_scores[metric] for elem in sub]
                    verification_scores[metric].loc[model, job_name] = best_all_scores[metric]
                verification_scores['roc'].loc[model, job_name] = best_roc
                verification_scores['n_top'].loc[model, job_name] = best_n_top

            self.plot_conf_matrix(best_conf_matrix, job_index + 1)

        return verification_scores

    def collect_scores(self, job_name, model) -> None:
        """Collect results over all seeds and bootstraps"""
        all_scores = {score: [] for score in self.metrics_to_collect}
        best_thresholds = [0.5] * self.n_bootstraps  # default threshold for binary classification
        recompute_scores = False  # whether to recompute scores with new threshold
        roc = []
        cum_conf_matrix = np.zeros(shape=(2, 2))
        for seed_index, seed in enumerate(self.seeds):
            try:
                scores = self.get_store('score', seed, job_name)[model]
            except KeyError:  # model not yet stored for this seed/job
                scores = {scoring: [] for scoring in self.metrics_to_collect}

            if scores[list(scores.keys())[0]]:  # else scores empty, i.e. not run for this job_name/n_top/seed
                # special metrics
                if self.use_youden_index:  # find best threshold and use it to compute other metrics
                    recompute_scores = True
                    for boot_iter in range(self.n_bootstraps):
                        fpr, tpr, thresholds = metrics.roc_curve(scores['true'][boot_iter], scores['probas'][boot_iter])
                        best_thresholds[boot_iter] = thresholds[np.argmax(tpr - fpr)]

                if 'roc' in self.metrics_to_collect:
                    for boot_iter in range(self.n_bootstraps):
                        fpr, tpr, thresholds = metrics.roc_curve(scores['true'][boot_iter], scores['probas'][boot_iter])
                        roc.append(
                            compute_roc_aucopt(
                                fpr,
                                tpr,
                                thresholds,
                                costs={'minoptsym': lambda fpr, tpr: -np.sqrt(fpr**2 + (1 - tpr) ** 2)},
                            )
                        )
                for boot_iter in range(self.n_bootstraps):  # compute cumulative confusion matrix
                    pred = np.where(np.array(scores['probas'][boot_iter]) >= best_thresholds[boot_iter], 1, 0)
                    cum_conf_matrix += metrics.confusion_matrix(scores['true'][boot_iter], pred)
                for score in self.metrics_to_collect:
                    if score == 'roc':  # already computed
                        continue
                    elif score == 'roc_auc_score':  # independent of threshold
                        all_scores[score].append(scores[score])
                        continue
                    if (
                        recompute_scores or score not in scores.keys() or len(scores[score]) < self.n_bootstraps
                    ):  # score needs to be recomputed or has not yet been computed
                        scores = self.compute_missing_scores(scores, score, best_thresholds)

                    all_scores[score].append(scores[score])
            else:
                np.delete(self.seeds, seed_index)

        mean_conf_matrix = cum_conf_matrix / (self.n_bootstraps * len(self.seeds))

        return all_scores, roc, mean_conf_matrix

    def compute_missing_scores(self, scores, score, thresholds):
        """Compute missing scores for a given metric and threshold"""
        try:  # try sklearn metrics
            scores[score] = [
                getattr(metrics, score)(
                    scores['true'][boot_iter],
                    np.where(np.array(scores['probas'][boot_iter]) >= thresholds[boot_iter], 1, 0),
                )
                for boot_iter in range(self.n_bootstraps)
            ]
        except AttributeError:  # try imbalanced learn metrics (e.g. for specificity)
            scores[score] = [
                getattr(imb_metrics, score)(
                    scores['true'][boot_iter],
                    np.where(np.array(scores['probas'][boot_iter]) >= thresholds[boot_iter], 1, 0),
                )
                for boot_iter in range(self.n_bootstraps)
            ]

        return scores

    def reduce_scores(self, scores, function):
        reduced_scores = {
            metric: pd.DataFrame(
                np.vectorize(function)(scores[metric]),
                index=scores[metric].index,
                columns=scores[metric].columns,
            )
            for metric in self.metrics_to_plot  # cannot reduce roc
        }
        return reduced_scores

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
                plt.savefig(os.path.join(self.report_dir, f'ROC_best_strat_{job_index+1}.{self.plot_format}'), dpi=300)
                plt.clf()

        roc_ax.set_title('Best mean ROC for all strategies')
        roc_plot.savefig(os.path.join(self.report_dir, f'ROC_best_per_strat.{self.plot_format}'), dpi=300)

    def plot_conf_matrix(self, conf_matrix, job_index):
        plt.figure()
        plt.tight_layout()
        conf_matrix = metrics.ConfusionMatrixDisplay(conf_matrix)
        conf_matrix.plot(cmap='Blues', values_format='.2f')
        plt.savefig(os.path.join(self.report_dir, f'confusion_matrix_strat_{job_index}.{self.plot_format}'), dpi=300)
        plt.clf()

    def plot_heatmaps(self, mean_scores):
        cmaps = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys', 'YlGnBu', 'YlOrRd', 'PuBu', 'PuRd']
        for i, score in enumerate(self.metrics_to_plot):
            fig = plt.figure()
            sns.heatmap(
                mean_scores[score],
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
            plt.savefig(
                os.path.join(self.report_dir, f'results_heatmap_{self.metrics_to_plot[i]}.{self.plot_format}'), dpi=300
            )
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


def collect_results() -> None:
    config = ConfigManager()(save=False)
    logger.remove()
    logger.add(sys.stderr, level=config.meta.logging_level)
    if config.meta.ignore_warnings:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    CollectResults(config)()
    logger.info('Results collected successfully.')


if __name__ == '__main__':
    collect_results()
