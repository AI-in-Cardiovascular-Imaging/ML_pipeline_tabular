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

        pred_function, x_train_norm, x_test_norm = self.verification(
            seed, 0, job_name, fit_imputer, model=[model], n_top_features=[n_top], explain_mode=True 
        )

        explainer = KernelShap(pred_function)
        explainer.fit(x_train_norm[features])
        explanation = explainer.explain(x_test_norm[features], feature_names=features)

        shap_plot, _ = plt.subplots()
        shap.summary_plot(explainer.shap_values[-1], x_test_norm[features], features)
        shap_plot.savefig(os.path.join(self.out_dir, f'KernelSHAP_{model}_strat_{job_index}.{self.plot_format}'))
