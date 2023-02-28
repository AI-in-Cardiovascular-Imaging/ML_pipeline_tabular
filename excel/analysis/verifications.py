from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from excel.analysis.utils.normalisers import Normaliser


class VerifyFeatures(Normaliser):
    """Train random forest classifier to verify feature importance"""

    def __init__(self, config, v_data, v_data_test=None, features=None):
        super().__init__()
        self.config = config
        self.target_label = config.analysis.experiment.target_label
        self.seed = config.analysis.run.seed
        self.oversample = config.analysis.run.verification.oversample
        self.models = config.analysis.run.verification.models
        self.param_grids = config.analysis.run.verification.param_grids

        if v_data_test is None:
            x, y = self.prepare_data(v_data, features_to_keep=features)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x,
                y,
                stratify=y,
                test_size=0.20,
                random_state=self.seed,
            )
        else:  # v_data already split in train and test set
            self.x_train, self.y_train = self.prepare_data(v_data, features_to_keep=features)
            self.x_test, self.y_test = self.prepare_data(v_data_test, features_to_keep=features)
            
        if self.oversample:
            oversampler = RandomOverSampler(random_state=self.seed)
            self.x_train, self.y_train = oversampler.fit_resample(self.x_train, self.y_train)

    def __call__(self):
        """Train random forest classifier to verify feature importance"""
        # clf = VotingClassifier(
        #     estimators=[
        #         ('gb', GradientBoostingClassifier(random_state=self.seed)),
        #         ('et', ExtraTreesClassifier(random_state=self.seed)),
        #         ('ab', RandomForestClassifier(random_state=self.seed)),
        #         ('rf', AdaBoostClassifier(random_state=self.seed)),
        #         ('bc', BaggingClassifier(random_state=self.seed)),
        #     ],
        #     voting='hard',
        # )
        clf = LogisticRegressionCV(scoring='average_precision', random_state=self.seed, class_weight='balanced')
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

    def prepare_data(self, data: pd.DataFrame, features_to_keep: list=None):
        y = data[self.target_label]
        data = self.z_score_norm(data)
        x = data.drop(
            columns=[c for c in data.columns if c not in features_to_keep], axis=1
        )  # Keep only selected features
        if self.target_label in x.columns: # ensure that target column is dropped
            x = x.drop(self.target_label, axis=1)

        return x, y
