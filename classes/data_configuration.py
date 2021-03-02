"""
Conjunto de classes para organizar os dados em formatos reconhecidos pelo banco de dados
"""

from __future__ import annotations
import numpy as np
import _pickle as pickle
import os
import mne


class Epochs:
    def __init__(self, x, classe: str, subject_name: str, data_type=None) -> None:
        # Epoca original de dados
        self.data = x
        # Classe do conjunto de epocas
        self.classe = classe
        # Nome do sujeito ao qual esta instância está associada
        self.subject_name = subject_name
        # data_type
        self.data_type = data_type
        # Pasta onde ficará este conjunto de epocas
        self.epc_folderpath = os.path.join("subject_files", subject_name, f"epochs_{data_type}")

        # bloco para verificar principalmente se há mais de uma matriz de epocas
        try:
            self.n_trials = self.data.shape[2]
        except IndexError:
            n_ch = self.data.shape[0]
            n_samp = self.data.shape[1]
            self.n_trials = 1
            self.data = self.data.reshape(n_ch, n_samp, 1)

    # Adiciona uma epoca no conjunto original de dados
    def add_epoch(self, new_data: Epochs):
        self.data = np.append(self.data, new_data.data, axis=2)
        try:
            self.n_trials += new_data.data.shape[2]
        except IndexError:
            self.n_trials += 1

    def save_epoch(self):
        try:
            if not os.path.exists(self.epc_folderpath):
                os.makedirs(self.epc_folderpath)

            with open(os.path.join(self.epc_folderpath, f"{self.classe}_epoch.pkl"), "wb") as file:
                pickle.dump(self, file, -1)

        except IOError as e:
            raise IOError(f"Não foi possível salvar a época: {e}")

    @classmethod
    def dict_from_subject_name(cls, sbj_name, data_type) -> dict:
        epochs = dict()
        path = os.path.join("subject_files", sbj_name, f"epochs_{data_type}")
        files = sorted(os.listdir(path))
        for filename in files:
            with open(os.path.join(path, filename), "rb") as file:
                epoch = pickle.load(file)
                epochs[epoch.classe] = epoch
        return epochs


class Headset:
    def __init__(self, name, ch_names, sfreq):
        """
        Inicialização e criação de uma nova instância de um headset, adotando um nome, um array com os nomes dos canais
        disponíveis nesse headset e a frequência de amostragem.

        Parameters
        ----------
        name
        ch_names
        sfreq
        """
        self.name = name
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.montage = self._set_montage()

    @classmethod
    def from_headset_name(cls, name: str):
        if not name.endswith(".pkl"):
            name += ".pkl"

        with open(f"headsets/{name}", "rb") as file:
            return pickle.load(file)

    def save_headset(self):
        name = self.name

        if not name.endswith(".pkl"):
            name += ".pkl"

        with open(f"headsets/{name}", "wb") as file:
            pickle.dump(self, file)

    # Cria um objeto de montagem de acordo com os canais informados
    def _set_montage(self):
        """
        Gera um objeto de montagem (MNE tools) para o headset da instância de acordo com os dados passados na iniciali-
        zação.
        """
        dataset_ch_names = self.ch_names

        # Carrega o arquivo
        file = np.loadtxt('headsets/plotting_1005.txt', dtype={
            'names': ['ch', 'x', 'y', 'z'],
            'formats': ['S6', 'f4', 'f4', 'f4']
        })

        # Cria a variavel que ira guardar o nome dos canais
        ch_nums = len(file)
        all_ch_names = []
        coord = np.zeros([342, 3])

        # Passa pelo arquivo linha por linha
        for ii, jj in enumerate(file):
            # Converte a primeira coluna para string e salva na lista
            all_ch_names.append(file[ii][0].decode('ascii'))

            # Converte as coordenadas para float e guarda na matriz
            for coo in range(3):
                coord[ii, coo] = float(file[ii][coo + 1]) / 10

        # Salva em uma matriz as posições de cada um dos canais rferenciados em 'names'
        ch_coord = coord[np.where([all_ch_names[i] in dataset_ch_names for i in range(ch_nums)])[0]]

        # Salva a posição do eletrodo Nasio para referencia
        nz_pos = coord[np.where([all_ch_names[i] in ['Nz'] for i in range(ch_nums)])[0]].reshape(3)
        # Salva a posição do eletrodo lpa para referencia
        lpa_pos = coord[np.where([all_ch_names[i] in ['LPA'] for i in range(ch_nums)])[0]].reshape(3)
        # Salva a posição do eletrodo rpa para referencia
        rpa_pos = coord[np.where([all_ch_names[i] in ['RPA'] for i in range(ch_nums)])[0]].reshape(3)

        # Cria o dicionario de montagem do layout
        ch_pos = {k: v for k, v in zip(dataset_ch_names, ch_coord)}

        # Cria o objeto de montagem do laout
        montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            nasion=nz_pos,
            lpa=lpa_pos,
            rpa=rpa_pos
        )

        return montage


class SubjectTimingConfigs:
    def __init__(self, trial_duration, sample_start, sample_end, ica_start, ica_end,
                 time_between=None, epc_size=None):
        """
        Inicialização de uma instância de configuração das marcações de tempo para as amostras de uma gravação. Esta
        classe é feita para acompanhar classes filhas de Subject, definindo o padrão das gravações de uma mesma pessoa.
        Aqui que o tamanho das Epocas são definidas, um conjunto de dados padronizados que servirão para amostrar os
        dados de uma mesma pessoa.

        Parameters
        ----------
        trial_duration: Tempo de duração de um ensaio, marcado pelos eventos de uma gravação
        sample_start: Instante de tempo do inicio da amostra útil do ensaio
        sample_end: Instante de tempo do fim da amostra útil do ensaio
        time_between: Caso seja possível coletar mais de uma epoca de uma amostra, indica o intervalo entre seus inicios
        epc_size: Tamanho fixo, em segundos, que deve ter uma época desde sujeito
        """
        self.trial_duration = trial_duration
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.ica_start = ica_start
        self.ica_end = ica_end
        self.time_between = time_between
        self.epc_size = epc_size or sample_end - sample_start
