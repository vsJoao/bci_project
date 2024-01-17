from abc import ABC

from classes.abstracts import OneVsOneFBCSP, OneVsAllFBCSP, Classifier
import os
import _pickle as pickle
from sklearn import svm
import numpy as np
from scipy.stats import mode
import tensorflow as tf
from keras.models import load_model as keras_load_model
from tensorflow.keras import datasets, layers, models
from classes.fbcsp import FBCSP


class OneVsOneLinearSVM(OneVsOneFBCSP):
    def predict(self, signal: np.ndarray):
        feature = self.generate_set_of_features_for_signal(signal)

        prediction = list()
        for clas, model in self.classifier_models.items():
            prediction.append(*model.predict(feature[clas].T))

        res, _ = mode(prediction)
        return int(res)

    @classmethod
    def load_from_subjectname(cls, sbj_name):
        file_path = os.path.join("subject_files", sbj_name, "classifiers", "one_vs_one", "linear_svm.pkl")
        with open(file_path, "rb") as file:
            classifier = pickle.load(file)
        return classifier

    @property
    def classifier_method_name(self) -> str:
        return "linear_svm"

    def _set_classsifiers(self):
        self.generate_fbcsp()
        self.generate_subject_train_features()

        self.classifier_models = dict()

        train_features = self.get_subject_train_features_as_dict()

        for clas, features in train_features.items():
            x_train = features[:, :-1]
            y_train = features[:, -1]

            svm_model = svm.LinearSVC(C=0.1, max_iter=5000, dual=True)
            svm_model.fit(x_train, y_train)

            self.classifier_models[clas] = svm_model


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

    def _set_classsifiers(self):
        # TODO: Adicionar check se ja foram gerados os sets de caracteristicas e os sets de csp
        self.generate_fbcsp()
        self.generate_subject_train_features()

        self.classifier_models = dict()

        train_features = self.get_subject_train_features_as_dict()
        for clas, features in train_features.items():
            x_train = features[:, :-1]
            y_train = features[:, -1]

            svm_model = svm.LinearSVC(C=0.1, max_iter=5000, dual=True)
            svm_model.fit(x_train, y_train)

            self.classifier_models[clas] = svm_model

    def predict(self, signal: np.ndarray):
        classes = self.classification_order
        classes_dict = self.subject.classes
        classes_inv = {i: j for j, i in classes_dict.items()}
        w_fbcsp = FBCSP.dict_from_subject_name(self.subject.foldername, "one_vs_all")

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


class ConvolutionalClassifier(Classifier):
    def __init__(self, subject, dropout_rate=0.4, epoch_training=50):
        self.dropout_rate = dropout_rate
        self.epoch_training = epoch_training
        super().__init__(subject)

    def _set_classsifiers(self):
        train_epochs = self.subject.get_epochs_as_dict("train")
        classes = self.subject.classes

        labels_train = np.array([])
        data_train = np.array([])

        for cls, epoch in train_epochs.items():
            labels_train = np.append(labels_train, np.repeat(cls, epoch.n_trials))

            data_smp = epoch.data
            reshaped_data = np.zeros([data_smp.shape[2], data_smp.shape[0], data_smp.shape[1]])
            for nn in range(data_smp.shape[2]):
                reshaped_data[nn] = data_smp[:, :, nn]

            try:
                data_train = np.append(data_train, reshaped_data, axis=0)
            except ValueError:
                data_train = reshaped_data

        test_epochs = self.subject.get_epochs_as_dict("test")

        labels_test = np.array([])
        data_test = np.array([])

        # Junta todas as amostras de um mesmo sujeito em um unico array e muda a indexação para o padrão do tensorflow
        for cls, epoch in test_epochs.items():
            labels_test = np.append(labels_test, np.repeat(cls, epoch.n_trials))

            data_smp = epoch.data
            reshaped_data = np.zeros([data_smp.shape[2], data_smp.shape[0], data_smp.shape[1]])
            for nn in range(data_smp.shape[2]):
                reshaped_data[nn] = data_smp[:, :, nn]

            try:
                data_test = np.append(data_test, reshaped_data, axis=0)
            except ValueError:
                data_test = reshaped_data

        # Embaralha os dados contidos nas epocas
        p = np.random.permutation(len(data_test))
        data_test = data_test[p]
        labels_test = labels_test[p]

        p = np.random.permutation(len(data_train))
        data_train = data_train[p]
        labels_train = labels_train[p]

        classes_dict = {i: j for j, i in classes.items()}
        labels_train = np.array([classes_dict[i] for i in labels_train]) - 1
        labels_test = np.array([classes_dict[i] for i in labels_test]) - 1

        from scipy import signal

        sos = signal.iirfilter(
            N=6, Wn=[1, 40], rs=20, btype='bandpass',
            output='sos', fs=self.subject.headset.sfreq, ftype='cheby2'
        )

        data_train = signal.sosfilt(sos, data_train, axis=2)
        data_test = signal.sosfilt(sos, data_test, axis=2)

        model = models.Sequential([
            layers.InputLayer(input_shape=(data_train.shape[1], data_train.shape[2], 1)),
            layers.Conv2D(25, (1, 6), padding='same', activation='selu'),
            layers.Conv2D(25, (1, 6), padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.selu),
            layers.AveragePooling2D((1, 3), (1, 2)),
            layers.Dropout(self.dropout_rate),
            layers.Conv2D(50, (1, 6), padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.selu),
            layers.AveragePooling2D((1, 3), (1, 2)),
            layers.Dropout(self.dropout_rate),
            layers.Conv2D(100, (1, 6), padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.selu),
            layers.AveragePooling2D((1, 3), (1, 2)),
            layers.Dropout(self.dropout_rate),
            layers.Conv2D(200, (1, 6), padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.selu),
            layers.AveragePooling2D((1, 3), (1, 2)),
            layers.Dropout(self.dropout_rate),
            layers.Flatten(),
            layers.Dense(len(classes)),
            layers.Activation(tf.nn.softmax),
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        data_test = data_test.reshape([data_test.shape[0], data_test.shape[1], data_test.shape[2], 1])
        data_train = data_train.reshape([data_train.shape[0], data_train.shape[1], data_train.shape[2], 1])

        max_signal = self.subject.headset.max_signal

        self.history = model.fit(
            data_train, labels_train, epochs=self.epoch_training,
            validation_data=(data_test, labels_test)
        )

        self.classifier_models = model

    @property
    def classifier_foldername(self) -> str:
        return "ConvolutionalClassifier"

    @property
    def classifier_method_name(self) -> str:
        return "ConvolutionalClassifier"

    def predict(self, signal: np.ndarray):

        signal = signal.reshape([1, signal.shape[0], signal.shape[1], 1])

        output = self.classifier_models.predict(signal, verbose=0)
        prediction = output[0].argmax() + 1

        return prediction

    @classmethod
    def load_from_subjectname(cls, sbj_name):
        folderpath = os.path.join(
            "subject_files", sbj_name, "classifiers", "ConvolutionalClassifier")

        with open(os.path.join(folderpath, "ConvolutionalClassifier.pkl"), "rb") as file:
            classifier = pickle.load(file)

        model = keras_load_model(os.path.join(folderpath, "model.keras"))
        classifier.classifier_models = model

        return classifier

    def save_classifier(self):
        """
        Salva a instancia atual com as configurações atuais na sua pasta de acordo com o sujeito e o método utilizado
        """
        os.makedirs(self.folderpath, exist_ok=True)

        self.classifier_models.save(os.path.join(self.folderpath, "model.keras"))
        del self.classifier_models

        with open(os.path.join(self.folderpath, self.classifier_method_name + ".pkl"), "wb") as file:
            pickle.dump(self, file)
