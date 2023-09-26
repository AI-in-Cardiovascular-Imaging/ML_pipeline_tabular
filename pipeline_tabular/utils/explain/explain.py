import os
import shap

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from alibi.explainers import KernelShap

from pipeline_tabular.run.run import Run


class Explain(Run):
    """Explainability methods to understand model decision process"""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.plot_format = config.meta.plot_format
        self.expl_out_dir = os.path.join(self.out_dir, self.experiment_name, 'explain')
        os.makedirs(self.expl_out_dir, exist_ok=True)

    def __call__(self, job_name, job_index, model, n_top, mean_split_index, seeds, n_bootstraps) -> None:
        if n_bootstraps == 1:  # i.e. no bootstrapping
            seed = seeds[mean_split_index]
            np.random.seed(seed)
            boot_seed = np.random.randint(low=0, high=2**32, size=1)  # re-generate seed of original run
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
        conf_matrix.plot()
        plt.savefig(os.path.join(self.expl_out_dir, f'confusion_matrix_strat_{job_index}.{self.plot_format}'))
        plt.clf()

        # KernelSHAP plots
        explainer = KernelShap(pred_function)
        explainer.fit(x_train_norm[features])
        explanation = explainer.explain(x_test_norm[features], feature_names=features)

        shap.summary_plot(explanation.shap_values[1], x_test_norm[features], features, show=False)
        plt.savefig(os.path.join(self.expl_out_dir, f'KernelSHAP_positive_class_strat_{job_index}.{self.plot_format}'))
        plt.clf()

        shap.summary_plot(explanation.shap_values, x_test_norm[features], features, show=False)
        plt.savefig(os.path.join(self.expl_out_dir, f'KernelSHAP_both_classes_strat_{job_index}.{self.plot_format}'))
