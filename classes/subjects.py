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
                        eve.astype("int32")
                    )
                except IOError:
                    os.makedirs(os.path.join('subject_files', sbj, fif_folder_name))
                    mne.write_events(
                        os.path.join('subject_files', sbj, fif_folder_name, f'{sbj}_{n}_eve.fif'),
                        eve.astype("int32")
                    )
            except IOError:
                print('Não foi possível salvar {}_{}_eve.fif'.format(sbj, str(n)))


# Cria uma classe especializada para tratar dos sujeitos do dataset kaya
class KayaDatasetSubject(Subject):
    def __init__(self, headset: Headset, classes: dict, foldername: str, time_configs):
        super().__init__(headset, classes, foldername, time_configs)

    def _create_fif_files_from_original_data(self):
        files = {
            "HaLTSubjectA1602236StLRHandLegTongue.mat",
            "HaLTSubjectA1603086StLRHandLegTongue.mat",
            "HaLTSubjectA1603106StLRHandLegTongue.mat",
            "HaLTSubjectB1602186StLRHandLegTongue.mat",
            "HaLTSubjectB1602256StLRHandLegTongue.mat",
            "HaLTSubjectB1602296StLRHandLegTongue.mat",
            "HaLTSubjectC1602246StLRHandLegTongue.mat",
            "HaLTSubjectC1603026StLRHandLegTongue.mat",
            "HaLTSubjectE1602196StLRHandLegTongue.mat",
            "HaLTSubjectE1602266StLRHandLegTongue.mat",
            "HaLTSubjectE1603046StLRHandLegTongue.mat",
            "HaLTSubjectF1602026StLRHandLegTongue.mat",
            "HaLTSubjectF1602036StLRHandLegTongue.mat",
            "HaLTSubjectF1602046StLRHandLegTongue.mat",
            "HaLTSubjectG1603016StLRHandLegTongue.mat",
            "HaLTSubjectG1603226StLRHandLegTongue.mat",
            "HaLTSubjectG1604126StLRHandLegTongue.mat",
            "HaLTSubjectH1607206StLRHandLegTongue.mat",
            "HaLTSubjectH1607226StLRHandLegTongue.mat",
            "HaLTSubjectI1606096StLRHandLegTongue.mat",
            "HaLTSubjectI1606286StLRHandLegTongue.mat",
            "HaLTSubjectJ1611216StLRHandLegTongue.mat",
            "HaLTSubjectK1610276StLRHandLegTongue.mat",
            "HaLTSubjectK1611086StLRHandLegTongue.mat",
            "HaLTSubjectL1611166StLRHandLegTongue.mat",
            "HaLTSubjectL1612056StLRHandLegTongue.mat",
            "HaLTSubjectM1611086StLRHandLegTongue.mat",
            "HaLTSubjectM1611176StLRHandLegTongue.mat",
            "HaLTSubjectM1611246StLRHandLegTongue.mat",
        }
        sbjs = {'K01': 'A',
                'K02': 'B',
                'K03': 'C',
                'K04': 'E',
                'K05': 'F',
                'K06': 'G',
                'K07': 'H',
                'K08': 'I',
                'K09': 'J',
                'K10': 'K',
                'K11': 'L',
                'K12': 'M'}

        sbj = sbjs[self.foldername]
        sbj_files = []
        for filename in files:
            if filename.startswith(f"HaLTSubject{sbj}"):
                sbj_files.append(filename)

        self._save_fif(sbj_files)

    def _save_fif(self, filenames: list):
        sbj = self.foldername
        ch_names = self.headset.ch_names
        n = 0

        for filename in filenames:
            n += 1
            try:
                local = os.path.join("subject_files", "dataset_kaya", filename)
                file = loadmat(
                    local, verify_compressed_data_integrity=True,
                    chars_as_strings=True, simplify_cells=True)
                data = file['o'].copy()
                print(f"{sbj} carregado com sucesso")
            except IOError:
                print(f"Não foi possível carregar {sbj}")
                return

            # Carrega os dados contidos na gravação analisada
            x = data['data'].T.copy()
            # Carrega a informação da frequencia de gravação
            sfreq = data['sampFreq']
            # coleta as marcações dos eventos da gravação
            eve = np.array([])
            pt_anterior = 0
            for index, pt in enumerate(data['marker']):
                if pt != pt_anterior and pt != 0:
                    try:
                        eve = np.append(eve, np.array([[index, 0, pt]]), 0)
                    except ValueError:
                        eve = np.array([[index, 0, pt]])
                pt_anterior = pt

            # Mapeia os eletrodos como eeg ou eog
            chanel = ['eeg'] * data['chnames'].shape[0]
            # Carrega a montagem dos sensores:
            mnt = self.headset.montage
            # Cria a informação de todos os arquivos
            info = mne.create_info(ch_names=ch_names,
                                   sfreq=sfreq,
                                   ch_types=chanel)

            # Separa a gravação entre uma parte para treino e outra parte para teste
            n_events = eve.shape[0]
            train_factor = 0.8
            sep_eve_index = int(np.floor(n_events * train_factor))
            sep_rec_index = eve[sep_eve_index, 0] - 100

            x_train = x[:, :sep_rec_index]
            eve_train = eve[:sep_eve_index, :]

            x_test = x[:, sep_rec_index:]
            eve_test = eve[sep_eve_index:, :]
            eve_test[:, 0] = eve_test[:, 0] - sep_rec_index

            raw_train = mne.io.RawArray(x_train, info).set_montage(mnt)
            raw_test = mne.io.RawArray(x_test, info).set_montage(mnt)

            # Salva os arquivos _raw.fif
            try:
                try:
                    raw_train.save(os.path.join('subject_files', sbj, "raw_fif_files_train", f'{sbj}_{n}_raw.fif'))
                    raw_test.save(os.path.join('subject_files', sbj, "raw_fif_files_test", f'{sbj}_{n}_raw.fif'))
                except IOError:
                    os.makedirs(os.path.join('subject_files', sbj, "raw_fif_files_train"))
                    os.makedirs(os.path.join('subject_files', sbj, "raw_fif_files_test"))
                    raw_train.save(os.path.join('subject_files', sbj, "raw_fif_files_train", f'{sbj}_{n}_raw.fif'))
                    raw_test.save(os.path.join('subject_files', sbj, "raw_fif_files_test", f'{sbj}_{n}_raw.fif'))
            except IOError:
                print('Não foi possível salvar {}_{}_raw.fif'.format(sbj, str(n)))

            # Salva os arquivos _eve.fif
            try:
                try:
                    mne.write_events(
                        os.path.join('subject_files', sbj, "raw_fif_files_train", f'{sbj}_{n}_eve.fif'),
                        eve_train.astype("int32")
                    )
                    mne.write_events(
                        os.path.join('subject_files', sbj, "raw_fif_files_test", f'{sbj}_{n}_eve.fif'),
                        eve_test.astype("int32")
                    )
                except IOError:
                    os.makedirs(os.path.join('subject_files', sbj, "raw_fif_files_train"))
                    os.makedirs(os.path.join('subject_files', sbj, "raw_fif_files_test"))
                    mne.write_events(
                        os.path.join('subject_files', sbj, "raw_fif_files_train", f'{sbj}_{n}_eve.fif'),
                        eve_train.astype("int32")
                    )
                    mne.write_events(
                        os.path.join('subject_files', sbj, "raw_fif_files_test", f'{sbj}_{n}_eve.fif'),
                        eve_test.astype("int32")
                    )
            except IOError:
                print('Não foi possível salvar {}_{}_eve.fif'.format(sbj, str(n)))

