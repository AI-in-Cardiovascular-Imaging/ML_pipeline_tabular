import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.ensemble as ensemble
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from excel.analysis.utils.normalisers import Normaliser


class VerifyFeatures(Normaliser):
    """Train random forest classifier to verify feature importance"""

    def __init__(self, config, v_data):
        super().__init__()
        self.config = config
        self.target_label = config.analysis.experiment.target_label

        v_data = self.l1_norm(v_data)

        features = [
            'epi_radial_strain',
            'endo_circumf_strain',
            'Diagn_nyha2or3or4',
            'epi_longit_strain',
            'endo_radial_strain',
            'age',
            'endo_longit_strain',
            'Diagn_nyha3or4',
            'global_radial_strain',
            'Diagn_nyha',
            'bmi',
            'mace',
        ]

        v_data = v_data.drop(columns=[c for c in v_data.columns if c not in features], axis=1)  # Keep only features

        without_target = v_data.drop(self.target_label, axis=1)  # Drop target label from data

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            without_target,
            v_data[self.target_label],
            stratify=v_data[self.target_label],
            test_size=0.20,
            random_state=self.config.analysis.run.seed,
        )

    def __call__(self):
        """Train random forest classifier to verify feature importance"""
        clf = ensemble.VotingClassifier(
            estimators=[
                ('gb', ensemble.GradientBoostingClassifier()),
                ('et', ensemble.ExtraTreesClassifier()),
                ('ab', ensemble.RandomForestClassifier()),
                ('rf', ensemble.AdaBoostClassifier()),
                ('bc', ensemble.BaggingClassifier()),
            ],
            voting='hard',
        )

        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)

        print('Accuracy', accuracy_score(self.y_test, y_pred, normalize=True))
        print(classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        plt.title('Confusion matrix')
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()
