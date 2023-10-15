from classes.abstracts import OneVsOneFBCSP, OneVsAllFBCSP
import os
import _pickle as pickle
from sklearn import svm
import numpy as np
from scipy.stats import mode


class OneVsOneLinearSVM(OneVsOneFBCSP):
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

            svm_model = svm.LinearSVC(C=0.1, max_iter=5000, dual=True)
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


class OneVsAllLinearSVM(OneVsAllFBCSP):
    @classmethod
    def load_from_subjectname(cls, sbj_name):
        file_path = os.path.join("subject_files", sbj_name, "classifiers", "one_vs_all", "linear_svm.pkl")
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

            svm_model = svm.LinearSVC(C=0.1, max_iter=5000, dual=True)
            svm_model.fit(x_train, y_train)

            self.classifier_models[clas] = svm_model

    def predict_feature(self, signal: np.ndarray):
        classes = self.classification_order
        classes_dict = self.subject.classes
        classes_inv = {i: j for j, i in classes_dict.items()}
        w_fbcsp = self.subject.get_fbcsp_dict("one_vs_all")

        for index, clas in enumerate(classes[:-1]):
            next_clas = classes[index+1]
            class_id = classes_inv[clas]
            w = w_fbcsp[f"{clas}{next_clas}"]
            features = w.fbcsp_feature(signal)

            model = self.classifier_models[f"{clas}{next_clas}"]

            prediction = model.predict(features.T)
            prediction = int(prediction)

            if prediction == class_id:
                return prediction

        return classes_inv[classes[-1]]

    def run_testing_classifier(self):
        epochs = self.subject.get_epochs_as_dict("test")
        classes_dict = self.subject.classes
        classes_dict_inv = {i: j for j, i in classes_dict.items()}

        prediction_list = list()
        real_list = list()

        for clas, epoch in epochs.items():
            for i in range(epoch.n_trials):
                data = epoch.data[:, :, i]
                prediction = self.predict_feature(data)
                prediction_list.append(prediction)
                real_list.append(classes_dict_inv[clas])

        compare = [real_list[i] == prediction_list[i] for i, j in enumerate(real_list)]
        hit_rate = np.mean(compare)

        return {
            "hit_rate": hit_rate,
            "real_classes": real_list,
            "predicted_classes": prediction_list
        }

