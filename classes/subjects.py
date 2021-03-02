from classes.abstracts import Subject
from classes.data_configuration import Headset
from scipy.io import loadmat
import numpy as np
import mne
import os


# Cria uma classe especializada para tratar dos sujeitos da iv bci competition
class IVBCICompetitionSubject(Subject):

    def __init__(self, headset: Headset, classes: dict, foldername: str, time_configs):
        super().__init__(headset, classes, foldername, time_configs)

    def _create_fif_files_from_original_data(self):
        self._save_fif(f"{self.foldername}T.mat", data_type="train")
        self._save_fif(f"{self.foldername}E.mat", data_type="test")

    # Formata o banco de dados do formato originalmente disponível para o formato .fif
    def _save_fif(self, filename, data_type="train"):
        sbj = self.foldername
        ch_names = self.headset.ch_names

        try:
            local = os.path.join("subject_files", sbj, "original_data", filename)
            file = loadmat(local)
            data = file['data']
            print(f"{sbj} carregado com sucesso")
        except IOError:
            print(f"Não foi possível carregar {sbj}")
            return

        for i in range(3, 9):  # Carrega as 6 (das 9) runs de interesse de cada pessoa
            # Carrega em variáveis as informações do dataset
            x = data[0][i][0][0][0].copy().transpose()
            trial = data[0][i][0][0][1].copy()
            y = data[0][i][0][0][2].copy()
            sfreq = data[0][i][0][0][3][0][0].copy()

            # Mapeia os eletrodos como eeg ou eog
            chanel = ['eeg'] * 22 + ['eog'] * 3
            # Carrega a montagem dos sensores:
            mnt = self.headset.montage
            # Cria a informação de todos os arquivos
            info = mne.create_info(ch_names=ch_names,
                                   sfreq=sfreq,
                                   ch_types=chanel)

            # Pre carrega um array para os eventos
            eve = np.zeros([48, 3])

            # Carrega os dados para criação dos arquivos de eventos
            eve[:, [0]] = trial
            eve[:, [2]] = y

            # Cria o objeto RawArray com os dados
            raw = mne.io.RawArray(x, info).set_montage(mnt)

            fif_folder_name = "raw_fif_files_train" if data_type == "train" else "raw_fif_files_test"
            n = i - 2

            # Salva os arquivos _raw.fif
            try:
                try:
                    raw.save(os.path.join('subject_files', sbj, fif_folder_name, f'{sbj}_{n}_raw.fif'))
                except IOError:
                    os.makedirs(os.path.join('subject_files', sbj, fif_folder_name))
                    raw.save(os.path.join('subject_files', sbj, fif_folder_name, f'{sbj}_{n}_raw.fif'))
            except IOError:
                print('Não foi possível salvar {}_{}_raw.fif'.format(sbj, str(n)))

            # Salva os arquivos _eve.fif
            try:
                try:
                    mne.write_events(
                        os.path.join('subject_files', sbj, fif_folder_name, f'{sbj}_{n}_eve.fif'),
                        eve
                    )
                except IOError:
                    os.makedirs(os.path.join('subject_files', sbj, fif_folder_name))
                    mne.write_events(
                        os.path.join('subject_files', sbj, fif_folder_name, f'{sbj}_{n}_eve.fif'),
                        eve
                    )
            except IOError:
                print('Não foi possível salvar {}_{}_eve.fif'.format(sbj, str(n)))
