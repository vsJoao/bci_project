from abc import ABC, abstractmethod
import os
import _pickle as pickle
from sklearn import svm
from scipy.stats import mode


class OneVsOneClassificator(ABC):
    def __init__(self, subject_name: str, train_features: dict):
        self.subject_name = subject_name
        self.classifier_models = dict()
        self.folderpath = os.path.join("subject_files", subject_name, "classifiers", "one_vs_one")
        os.makedirs(self.folderpath, exist_ok=True)

        self._set_classsifiers(train_features)
        assert self.classifier_models

    @property
    @abstractmethod
    def classifier_method_name(self) -> str:
        pass

    @abstractmethod
    def _set_classsifiers(self, train_features):
        pass

    @abstractmethod
    def predict_feature(self, feature_dict: dict) -> int:
        pass

    @classmethod
    @abstractmethod
    def load_from_subjectname(cls, sbj_name):
        pass

    def save_classifier(self):
        with open(os.path.join(self.folderpath, self.classifier_method_name+".pkl"), "wb") as file:
            pickle.dump(self, file)


class OneVsOneLinearSVM(OneVsOneClassificator):
    @classmethod
    def load_from_subjectname(cls, sbj_name):
        with open(os.path.join("subject_files", sbj_name, "classifiers", "one_vs_one", "linear_svm.pkl"), "rb") as file:
            classifier = pickle.load(file)
        return classifier

    @property
    def classifier_method_name(self) -> str:
        return "linear_svm"

    def _set_classsifiers(self, train_features):
        for clas, features in train_features.items():
            x_train = features[:, :-1]
            y_train = features[:, -1]

            svm_model = svm.LinearSVC(C=0.1, max_iter=5000)
            svm_model.fit(x_train, y_train)

            self.classifier_models[clas] = svm_model

    def predict_feature(self, feature_dict: dict):
        prediction = list()
        for clas, model in self.classifier_models.items():
            prediction.append(*model.predict(feature_dict[clas].T))

        res, _ = mode(prediction)
        return int(res)
