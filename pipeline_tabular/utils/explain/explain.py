import os
import shap

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from alibi.explainers import KernelShap

from pipeline_tabular.data_handler.data_handler import DataHandler
from pipeline_tabular.utils.data_split import DataSplit
from pipeline_tabular.utils.imputers import Imputer
from pipeline_tabular.utils.normalisers import Normalisers
from pipeline_tabular.utils.verifications.verification import Verification
from pipeline_tabular.utils.helpers import generate_seeds


class Explain(DataHandler, Normalisers):
    """Explainability methods to understand model decision process"""

    def __init__(self, config) -> None:
        super().__init__()
        self.out_dir = config.meta.output_dir
        self.plot_format = config.meta.plot_format
        self.oversample = config.data_split.oversample
        self.jobs = config.selection.jobs
        self.data_split = DataSplit(config)
        self.imputation = Imputer(config)
        self.verification = Verification(config)

    def __call__(self, experiment_name, scores, opt_scoring, job_names, best_models, seeds, n_bootstraps) -> None:
        self.expl_out_dir = os.path.join(self.out_dir, experiment_name, 'explain')
        os.makedirs(self.expl_out_dir, exist_ok=True)
        self.load_frame(os.path.join(self.out_dir, experiment_name))

        for job_index, job_name in enumerate(job_names):
            best_model = best_models[job_name]
            n_top = scores['n_top'].loc[best_model, job_name]
            seed, boot_seed = self.get_seeds(scores, opt_scoring, job_name, best_model, seeds, n_bootstraps)
            self.data_split(seed, boot_seed)  # re-build data splits
            fit_imputer = self.imputation(seed)
            train_frame = self.get_store('frame', seed, 'train')
            if self.oversample:
                train_frame = self.over_sampling(train_frame, seed)
            norm = [step for step in self.jobs[job_index] if 'norm' in step][0]  # normalise data
            train_frame, _ = getattr(self, norm)(train_frame)
            self.set_store('frame', seed, 'train', train_frame)
            features = self.get_store('feature', seed, job_name, boot_iter=0)[:n_top]
            pred_function, estimator, x_train_norm, x_test_norm = self.verification(
                seed, 0, job_name, fit_imputer, model=[best_model], n_top_features=[n_top], explain_mode=True
            )
            self.plot_kernel_shap(pred_function, x_train_norm, x_test_norm, features, job_index + 1)
            if best_model == 'logistic_regression':
                coefficients = estimator.coef_
                self.plot_coefficients(coefficients, features, job_index + 1)

    def get_seeds(self, scores, opt_scoring, job_name, best_model, seeds, n_bootstraps):
        if n_bootstraps == 1:  # i.e. no bootstrapping
            opt_scores = scores[f'{opt_scoring}_score'].loc[best_model, job_name]
            mean_opt_score = np.mean(opt_scores)
            mean_split_index = np.argmin(
                np.abs(opt_scores - mean_opt_score)
            )  # find data split representative of mean model performance
            seed = seeds[mean_split_index]
            np.random.seed(seed)
            boot_seed = generate_seeds(seed, n_seeds=1)  # re-generate seed of original run
        else:
            raise NotImplementedError

        return seed, boot_seed

    def plot_kernel_shap(self, pred_function, x_train_norm, x_test_norm, features, job_index):
        explainer = KernelShap(pred_function)
        explainer.fit(x_train_norm[features])
        explanation = explainer.explain(x_test_norm[features], feature_names=features)
        shap.summary_plot(explanation.shap_values[1], x_test_norm[features], features, show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.expl_out_dir, f'KernelSHAP_positive_class_strat_{job_index}.{self.plot_format}'), dpi=300
        )
        plt.clf()
        shap.summary_plot(explanation.shap_values, x_test_norm[features], features, show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.expl_out_dir, f'KernelSHAP_both_classes_strat_{job_index}.{self.plot_format}'), dpi=300
        )
        plt.clf()

        heatmap_explainer = shap.KernelExplainer(
            lambda x: pred_function(x)[:, 1], x_train_norm[features]
        )  # need different format for heatmap plot
        values = heatmap_explainer(x_test_norm[features])
        shap.plots.heatmap(values, show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.expl_out_dir, f'KernelSHAP_heatmap_strat_{job_index}.{self.plot_format}'), dpi=300
        )
        plt.clf()

    def plot_coefficients(self, coefficients, features, job_index):
        plt.figure()
        plt.barh(features, coefficients[0], color='yellowgreen')
        plt.title('Feature coefficients from logistic regression')
        plt.tight_layout()
        plt.savefig(os.path.join(self.expl_out_dir, f'coefficients_strat_{job_index}.{self.plot_format}'), dpi=300)
        plt.clf()
