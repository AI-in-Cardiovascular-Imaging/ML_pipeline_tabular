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
        self.plot_format = config.meta.plot_format
        self.oversample = config.data_split.oversample
        self.jobs = config.selection.jobs
        self.data_split = DataSplit(config)
        self.imputation = Imputer(config)
        self.verification = Verification(config)

    def __call__(self, out_dir, job_name, job_index, model, n_top, mean_split_index, seeds, n_bootstraps) -> None:
        self.load_frame(out_dir)
        self.expl_out_dir = os.path.join(out_dir, 'explain')
        os.makedirs(self.expl_out_dir, exist_ok=True)
        if n_bootstraps == 1:  # i.e. no bootstrapping
            seed = seeds[mean_split_index]
            np.random.seed(seed)
            boot_seed = generate_seeds(seed, n_seeds=1)  # re-generate seed of original run
        else:
            raise NotImplementedError

        self.data_split(seed, boot_seed)  # re-build data splits
        fit_imputer = self.imputation(seed)
        train_frame = self.get_store('frame', seed, 'train')
        if self.oversample:
            train_frame = self.over_sampling(train_frame, seed)
        norm = [step for step in self.jobs[job_index] if 'norm' in step][0]  # normalise data
        train_frame, _ = getattr(self, norm)(train_frame)
        self.set_store('frame', seed, 'train', train_frame)
        features = self.get_store('feature', seed, job_name, boot_iter=0)[:n_top]
        pred_function, conf_matrix, x_train_norm, x_test_norm = self.verification(
            seed, 0, job_name, fit_imputer, model=[model], n_top_features=[n_top], explain_mode=True
        )

        # confusion matrix plot
        plt.figure()
        plt.tight_layout()
        conf_matrix.plot(cmap='Blues', values_format='d')
        plt.savefig(os.path.join(self.expl_out_dir, f'confusion_matrix_strat_{job_index}.{self.plot_format}'))
        plt.clf()

        # KernelSHAP plots
        explainer = KernelShap(pred_function)
        explainer.fit(x_train_norm[features])
        explanation = explainer.explain(x_test_norm[features], feature_names=features)
        shap.summary_plot(explanation.shap_values[1], x_test_norm[features], features, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.expl_out_dir, f'KernelSHAP_positive_class_strat_{job_index}.{self.plot_format}'))
        plt.clf()
        shap.summary_plot(explanation.shap_values, x_test_norm[features], features, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.expl_out_dir, f'KernelSHAP_both_classes_strat_{job_index}.{self.plot_format}'))
        plt.clf()

        heatmap_explainer = shap.KernelExplainer(
            lambda x: pred_function(x)[:, 1], x_train_norm[features]
        )  # need different format for heatmap plot
        values = heatmap_explainer(x_test_norm[features])
        shap.plots.heatmap(values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.expl_out_dir, f'KernelSHAP_heatmap_strat_{job_index}.{self.plot_format}'))
        plt.clf()
