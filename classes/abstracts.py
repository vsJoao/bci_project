from __future__ import annotations
import os
import pickle as pickle
from abc import ABC, abstractmethod
from itertools import combinations

import mne
import numpy as np

from classes.data_configuration import Headset, SubjectTimingConfigs, Epochs
from utils.artifact_remove import artifact_remove


class Subject(ABC):
    """
    Cada sudeito é identificado pelo nome da pasta dos seus arquivos contendo gravações, informações, vetores, matrizes
    de projeção espacial e amostras em geral. Um ficheiro é destinado às informações de uma única pessoa e cada pessoa
    deve ter um headset específico configurado para ela, caso desejar usar outro headset na mesma pessoa, deve-se criar
    outro ficheiro para isso, já que mudar o headset muda totalmente o conjunto de dados.
    Esta classe serve apenas como uma padronização para os tipos de sujeitos diferentes, considerando as classes que
    são analisadas, o headset, e o conjunto de dados.
    Uma pasta de sujeito deve ficar dentro da pasta "subject_files" e contém a seguinte estrutura:
    |nome_do_sujeito
      |original_data            # pasta contendo os dados originais
      |raw_fif_files_train      # pasta contendo os dados de treino no formato fif
      |raw_fif_files_test       # pasta contendo os dados de teste no formato fif
      |subject_info.pkl         # instância desta classe contendo dados do sujeito
    """
    def __init__(self, headset: Headset, classes: dict, foldername: str, time_configs: SubjectTimingConfigs):
        """
        Inicialização de uma instância da classe geral de sujeitos. O mínimo que deve conter é uma pasta, com os arqui-
        vos do sujeito já posicionados nessa localização, um dicionário referente às classes do sujeito e um objeto de
        Headset, informando as características do headset conectado

        Parameters
        ----------
        headset: Objeto com as informações do headset que foi utilizado nesse sujeito
        classes: Classes de movimentos analisados nesse sujeito
        foldername: Nome da pasta que identifica este sujeito
        time_configs: Objeto com as configurações de tempo utilizadas nas gravações deste sujeito
        """
        assert os.path.exists(f"subject_files/{foldername}")
        self.foldername = foldername
        self.headset = headset
        self.classes = classes
        self.time_config = time_configs

    def save_object(self):
        with open(f"subject_files/{self.foldername}/subject_info.pkl", "wb") as output_file:
            pickle.dump(self, output_file, -1)

    @classmethod
    def load_from_foldername(cls, sbj_folder) -> Subject:
        with open(f"subject_files/{sbj_folder}/subject_info.pkl", "rb") as file:
            return pickle.load(file)

    def set_fif_files(self, *args, **kwargs):
        """
        Gera um conjunto de dados no foramto que é trabalhado por este programa a partir das gravações no formato
        que foram originalmente gravadas. Ao fim, faz uma checagem para verificar se os arquivos foram salvos correta-
        mente.

        Parameters
        ----------
        args: Parametros a serem definidos de acordo com a função que faz a formatação dos dados.
        kwargs: Parametros a serem definidos de acordo com a função que faz a formatação dos dados.
        """
        self._create_fif_files_from_original_data(*args, **kwargs)

        # Adiciona uma checagem para verificar se os arquivos foram gerados corretamente
        for d_type in ["train", "test"]:
            path = os.path.join("subject_files", self.foldername, f"raw_fif_files_{d_type}")
            n = self.get_recording_qtd(d_type)
            for i in range(1, n + 1):
                raw_path = os.path.join(path, f"{self.foldername}_{i}_raw.fif")
                eve_path = os.path.join(path, f"{self.foldername}_{i}_eve.fif")
                if not (os.path.exists(raw_path) and os.path.exists(eve_path)):
                    raise FileNotFoundError(f"Subject .fif file not found for {d_type} and id {i}")

    @abstractmethod
    def _create_fif_files_from_original_data(self, *args, **kwargs) -> None:
        """
        Esta função deve acessar a pasta "original_data" e organizar os dados no formato .fif (dentro da pasta correta),
        decidindo desde já como realizar a separação entre os conjuntos de treino e de teste para o banco de dados. Aqui
        também devem ser salvos os arquivos de eventos, indicando o ínicio de cada amostra rotulada pelo evento. Os ar-
        quivos gerados obrigatóriamente devem estar nomeados como
        "{nome_da_pasta}_{numero_identificador_da_gravação}_'raw'/'eve'.fif"
        sendo escolhido o sufixo de arquivos raw ou de eventos de acordo com o próprio arquivo.
        """
        pass

    def pick_fif_files(self, data_type, recording_id):
        """
        Carrega arquivos selecionados pelo identificador e pelo tipo (teste ou treino), deixando os dados pré-carregados
        e os retornando para manipulação

        Parameters
        ----------
        data_type: Tipo de dado, pode admitir os valores "test" ou "train"
        recording_id: Identificador da gravação dentre as diversas existentes no formato .fif

        Returns
        -------
        raw: Os dados da gravação no formato mne.RawArray
        eve: A matriz de eventos que marca a gravação
        """
        assert data_type == "test" or data_type == "train"

        events_filename = os.path.join(
            "subject_files", self.foldername, f"raw_fif_files_{data_type}", f"{self.foldername}_{recording_id}_eve.fif"
        )
        raw_filename = os.path.join(
            "subject_files", self.foldername, f"raw_fif_files_{data_type}", f"{self.foldername}_{recording_id}_raw.fif"
        )

        try:
            eve = mne.read_events(events_filename)
            raw = mne.io.read_raw_fif(raw_filename, preload=True)
        except FileNotFoundError:
            raise FileNotFoundError(f".fif files does not exists for {self.foldername} subject")

        return raw, eve

    def get_recording_qtd(self, data_type):
        """
        Identifica a quantidade de gravações que têm na pasta desse sujeiro referido pela instância

        Parameters
        ----------
        data_type: Admite os valores "train" ou "test" para definir de qual ficheiro serão lidas as gravações

        Returns
        -------
        last_id: O id da última gravação, que deve marcar também a quantidade de grvações na pasta
        """
        filenames = sorted(os.listdir(f"subject_files/{self.foldername}/raw_fif_files_{data_type}"))
        last_file = filenames[-1]
        last_id = last_file.split("_")[-2]

        return int(last_id)

    def _get_sample_from_trial(self, signal: mne.io.RawArray, apply_ica=True):
        """
        De acordo com as confugirações de marcação de tempo de uma trial, separa o trecho de informção útil no
        formato de uma epoca e retorna essa pequena amostra como resultado.

        Parameters
        ----------
        signal: Uma trial já separada de uma gravação que dentro dela encontra-se intervalos com informação útil
        apply_ica: Determina se o sinal irá passar pela limpeza.

        Returns
        -------
        new_epc: A amostra extraída de dentro da epoca.
        """
        ch = signal.pick('eeg').info['nchan']
        n_samp = int((self.time_config.sample_end - self.time_config.sample_start) * signal.info["sfreq"] + 1)

        # Realiza a remoção de artefatos
        if apply_ica:
            raw_clean, flag = artifact_remove(signal)
        else:
            raw_clean = signal.copy()

        # Salva a epoca
        new_epc = \
            raw_clean.crop(tmin=self.time_config.sample_start, tmax=self.time_config.sample_end).\
            get_data().reshape(ch, n_samp, 1)

        return new_epc

    def _generate_epochs_from_recording(self, data_type, recording_id):
        """
        Função que analisa uma gravação e cria um dicionário com instâncias de épocas (intervalos de interesse) desta
        gravação única, com chaves para cada classe de sinal utilizada neste sujeito.

        Parameters
        ----------
        data_type: Identifica o tipo da gravação
        recording_id: Indica o id da gravação que será analisada

        Returns
        -------
        x: Dicionário de Epocas, sendo uma chave para as Epocas de cada classe
        """
        raw, eve = self.pick_fif_files(data_type, recording_id)

        # Pré aloca um dicionário que será utilizado como retorno da função
        x = dict()

        # Esse laço roda cada uma das trials dentro de um arquivo
        for n, i in enumerate(eve[:, 0] / raw.info["sfreq"]):

            if eve[n, 2] not in self.classes:
                continue

            # Salva a classe de movimento atual
            class_mov = self.classes[eve[n, 2]]

            # Coleta uma amostra de (ica_end - ica_start) segundos para análise
            raw_samp = raw.copy().pick('eeg'). \
                crop(tmin=i+self.time_config.ica_start, tmax=i+self.time_config.ica_end)

            # TODO: Aplicar nesse ponto a funcionalidade de extrair mais de uma época por trial
            new_epc = self._get_sample_from_trial(raw_samp, apply_ica=True)

            # Adiciona o sinal atual em sua respectiva classe do dicionário X
            try:
                x[class_mov].add_epoch(new_epc)
            except KeyError:
                x[class_mov] = Epochs(
                    x=new_epc,
                    classe=class_mov,
                    subject_name=self.foldername,
                    data_type=data_type
                )

        return x

    def get_epoch(self, data_type, clas):
        return np.load(
            os.path.join("subject_files", self.foldername, f"epochs_{data_type}", f"{clas}_epoch.npy"),
            allow_pickle=True
        ).item()

    def get_epochs_as_dict(self, data_type):
        epochs = Epochs.dict_from_subject_name(self.foldername, data_type)
        return epochs

    def generate_epochs(self):
        for d_type in ["train", "test"]:
            n_of_files = self.get_recording_qtd(d_type)
            epc_dict = dict()

            for file_id in range(1, n_of_files+1):
                epc_dict_temp = self._generate_epochs_from_recording(d_type, file_id)

                for clas in epc_dict_temp:
                    if file_id == 1:
                        epc_dict[clas] = epc_dict_temp[clas]
                    else:
                        epc_dict[clas].add_epoch(epc_dict_temp[clas])

            for clas, epc in epc_dict.items():
                epc.save_epoch()

    def get_fbcsp_dict(self, set_type) -> dict:
        from classes.fbcsp import FBCSP
        return FBCSP.dict_from_subject_name(self.foldername, set_type)


class OneVsOneClassificator(ABC):
    """
    Uma abstração para modelos de classificadores que funcionem utilizando o padrão 1 vs 1, comparando todas as combina-
    ções possíveis de classes, mas que pode utilizar diferentes metodologias para a separação dos vetores, como svm,
    deep learning, multi-layer perceptron, random-forest, KNN...
    Ainda é preciso criar novos métodos
    """
    def __init__(self, subject: Subject):
        self.subject = subject
        self.classifier_models = dict()
        self.folderpath = os.path.join("subject_files", subject.foldername, "classifiers", "one_vs_one")

        self._fit_classifier_models()

    def _fit_classifier_models(self):
        self.generate_fbcsp()
        self.generate_subject_train_features()
        self.generate_subject_test_features()

        train_features = self.get_subject_train_features_as_dict()
        self._set_classsifiers(train_features)
        assert self.classifier_models

    def save_classifier(self):
        """
        Salva a instancia com as configurações atuais na sua pasta de acordo com o sujeito e o método utilizado
        """
        os.makedirs(self.folderpath, exist_ok=True)
        with open(os.path.join(self.folderpath, self.classifier_method_name+".pkl"), "wb") as file:
            pickle.dump(self, file)

    def generate_fbcsp(self, fb_freqs=None, m: int = 2):
        """
        Gera as matrizes de projeção espacial para separação das classes duas a duas, no modelo de classificação um
        contra um. Esse formato faz uma combinação dentre todas as classes existentes para ser capaz de separar todas
        entre elas mesmas.

        Parameters
        ----------
        fb_freqs: dict
            Dicionário contendo todas as bandas de frequencias nas quais o sinal original será filtrado antes de gerar
            as matrizes de projeção espacial
        m: int
            Quantidade de sinais que serão aproveitados na matriz resultante do sinal após sofrer a transformação
            realizada pelas matrizes de separação.
        """
        from classes.fbcsp import FBCSP

        # O ideal é que o usuário defina as bandas de frequencia, mas caso não, essas faixas são as pré definidas
        # para todos os usuários que utilizarem esse algorítmo
        if fb_freqs is None:
            fb_freqs = {1: [8, 12], 2: [12, 16], 3: [16, 20], 4: [20, 24], 5: [24, 28], 6: [28, 32]}

        classes_names = list(self.subject.classes.values())
        epochs = self.subject.get_epochs_as_dict("train")

        for i, j in combinations(classes_names, 2):
            w = FBCSP(epochs[i], epochs[j],
                      subject_name=self.subject.foldername, fs=self.subject.headset.sfreq, m=m, filterbank=fb_freqs)
            w.save_fbcsp(set_type="one_vs_one")

    def generate_subject_train_features(self):
        """
        Gera os vetores de características de treinamento para o formato de classificação um contra um.
        Os vetores de separação para cada combinação de duas classes diferentes são salvas em arquivos separados dentro
        da sua respectiva pasta.
        """
        classes_names = list(self.subject.classes.values())
        w_fbcsp = self.subject.get_fbcsp_dict("one_vs_one")
        epc_dict = self.subject.get_epochs_as_dict("train")
        path = os.path.join("subject_files", self.subject.foldername, "features_train", "one_vs_one")

        for i, j in combinations(classes_names, 2):
            f = w_fbcsp[f"{i}{j}"].generate_features_from_epochs(
                epc_dict[f"{i}"], epc_dict[f"{j}"], self.subject.classes
            )
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, f"{i}{j}_feature.npy"), f)

    def get_subject_train_features_as_dict(self):
        """
        Carrega o conjunto de treino desse sujeito dentro de um dicionário que utiliza as classes como chaves.

        Returns
        -------
        f: dict
            Dicionário de classes de movimento, cada qual com seu conjunto de vetores de características.
        """
        f = dict()
        path = os.path.join("subject_files", self.subject.foldername, "features_train", "one_vs_one")

        for i, j in combinations(self.subject.classes.values(), 2):
            f[f"{i}{j}"] = np.load(os.path.join(path, f"{i}{j}_feature.npy"))

        return f

    def generate_set_of_features_for_signal(self, signal: np.ndarray):
        """
        Gera um dicionário de vetores de características para um sinal de classes desconhecida para cada uma das
        combinações de separação de classes.

        Parameters
        ----------
        signal: Um sinal dentro de um array numpy MxN, onde M é a quantidade de canais e N é a quantidade de pontos.

        Returns
        -------
        f: dicionário com um vetor de características para cada combinação possível de separação de classes.
        """
        w_fbcsp = self.subject.get_fbcsp_dict("one_vs_one")
        f = dict()

        for clas, w in w_fbcsp.items():
            f[clas] = w.fbcsp_feature(signal)

        return f

    def generate_subject_test_features(self):
        epc_dict = self.subject.get_epochs_as_dict("test")
        path = os.path.join("subject_files", self.subject.foldername, "features_test", "one_vs_one")
        test_features = list()

        for clas, epc in epc_dict.items():
            for i in range(epc.n_trials):
                f_temp = self.generate_set_of_features_for_signal(epc.data[:, :, i])
                test_features.append({"feature": f_temp, "class": clas})

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, "test_features.npy"), test_features)

    def get_subject_test_features(self):
        path = os.path.join("subject_files", self.subject.foldername, "features_test", "one_vs_one")
        features = np.load(os.path.join(path, "test_features.npy"), allow_pickle=True)
        return features

    @abstractmethod
    def predict_feature(self, feature_dict: dict): pass

    @abstractmethod
    def _set_classsifiers(self, train_features): pass

    @classmethod
    @abstractmethod
    def load_from_subjectname(cls, sbj_name): pass

    @abstractmethod
    def run_testing_classifier(self): pass

    @property
    @abstractmethod
    def classifier_method_name(self) -> str: pass
