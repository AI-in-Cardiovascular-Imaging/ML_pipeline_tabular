import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.ensemble as ensemble
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from excel.analysis.utils.normalisers import Normaliser


class VerifyFeatures(Normaliser):
    """Train random forest classifier to verify feature importance"""

    def __init__(self, config, v_data, features):
        super().__init__()
        self.config = config
        self.target_label = config.analysis.experiment.target_label
        self.seed = config.analysis.run.seed
        self.oversample = config.analysis.run.oversample

        y = v_data[self.target_label]
        v_data = self.z_score_norm(v_data)
        x = v_data.drop(columns=[c for c in v_data.columns if c not in features], axis=1)  # Keep only selected features

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            stratify=y,
            test_size=0.20,
            random_state=self.seed,
        )

        if self.oversample:
            oversampler = RandomOverSampler(random_state=self.seed)
            self.x_train, self.y_train = oversampler.fit_resample(self.x_train, self.y_train)

    def __call__(self):
        """Train random forest classifier to verify feature importance"""
        clf = ensemble.VotingClassifier(
            estimators=[
                ('gb', ensemble.GradientBoostingClassifier(random_state=self.seed)),
                ('et', ensemble.ExtraTreesClassifier(random_state=self.seed)),
                ('ab', ensemble.RandomForestClassifier(random_state=self.seed)),
                ('rf', ensemble.AdaBoostClassifier(random_state=self.seed)),
                ('bc', ensemble.BaggingClassifier(random_state=self.seed)),
            ],
            voting='hard',
        )

        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)

        print('Accuracy', accuracy_score(self.y_test, y_pred, normalize=True))
        print('Average precision', average_precision_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        plt.figure(figsize=(10, 7))
        plt.title('Confusion matrix')
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        # plt.show()
