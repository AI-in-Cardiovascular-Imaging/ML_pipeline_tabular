import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import sklearn.ensemble as ensemble
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from excel.analysis.utils.normalisers import Normaliser


class VerifyFeatures(Normaliser):
    """Train random forest classifier to verify feature importance"""

    def __init__(self, config, v_data):
        super().__init__()
        self.config = config
        self.target_label = config.analysis.experiment.target_label

        v_data = self.l2_norm(v_data)

        features = ['fhxcad___0', 'Diagn_sympt_card_arrest', 'Diagn_sympt_edema', 'global_radial_strain',
                    'global_circumf_strain', 'endo_circumf_strain', 'epi_longit_strain', 'epi_circumf_strain',
                    'baseline_meds_diuretic', 'global_longit_strain', 'epi_radial_strain', 'Diagn_nyha3or4',
                    'endo_longit_strain', 'endo_radial_strain', 'Diagn_nyha', 'bmi', 'age', 'mace']

        v_data = v_data.drop(columns=[c for c in v_data.columns if c not in features], axis=1)

        test_data = v_data.sample(frac=0.3, random_state=self.config.analysis.run.seed)
        train_data = v_data.drop(test_data.index)

        logger.info('v_data', v_data.shape)
        logger.info('train_data', train_data.shape)
        logger.info('test_data', test_data.shape)

        self.x_train = train_data.drop(self.config.analysis.experiment.target_label, axis=1)
        self.y_train = train_data[self.config.analysis.experiment.target_label]

        self.x_test = test_data.drop(self.config.analysis.experiment.target_label, axis=1)
        self.y_test = test_data[self.config.analysis.experiment.target_label]

    def __call__(self):
        """Train random forest classifier to verify feature importance"""
        clf = ensemble.GradientBoostingClassifier(random_state=self.config.analysis.run.seed)
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)

        print('Accuracy', accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        plt.title('Confusion matrix')
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()
