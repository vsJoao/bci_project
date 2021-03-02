from classes.abstracts import OneVsOneClassificator
import os
import _pickle as pickle
from sklearn import svm
import numpy as np
from scipy.stats import mode


class OneVsOneLinearSVM(OneVsOneClassificator):
    @classmethod
    def load_from_subjectname(cls, sbj_name):
        file_path = os.path.join("subject_files", sbj_name, "classifiers", "one_vs_one", "linear_svm.pkl")
        with open(file_path, "rb") as file:
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

    def run_testing_classifier(self):
        features_test_list = self.get_subject_test_features()
        compare = list()

        inverted_clas_dict = dict(zip(self.subject.classes.values(), self.subject.classes.keys()))
        real_classes = list()
        prediction_list = list()

        for feature in features_test_list:
            res = self.predict_feature(feature["feature"])
            compare.append(self.subject.classes[res] == feature["class"])

            real_classes.append(inverted_clas_dict[feature["class"]])
            prediction_list.append(res)

        hit_rate = np.mean(compare)

        return {
            "hit_rate": hit_rate,
            "real_classes": real_classes,
            "predicted_classes": prediction_list
        }
