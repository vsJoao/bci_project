from classes.subjects import *
from classes.classifications import OneVsOneLinearSVM, OneVsAllLinearSVM, ConvolutionalClassifier
from classes.data_configuration import SubjectTimingConfigs
from classes.data_configuration import Headset


# Criação do objeto de temporização do sinal
timing_configs = SubjectTimingConfigs(
    trial_duration=7.5, sample_start=3.5, sample_end=6, ica_start=0, ica_end=7, epc_size=None, time_between=None
)
# timing_configs = SubjectTimingConfigs(
#     trial_duration=7.5, sample_start=3.5, sample_end=4.3, ica_start=0, ica_end=7, epc_size=None, time_between=None
# )

# Canais que são utilizados no dataset
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
            'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
            'EOG1', 'EOG2', 'EOG3']

# Criação do objeto do headset que foi feita a gravação
headset = Headset(ch_names=ch_names, max_signal=100, sfreq=250, name="iv_bci_headset")
headset.save_headset()

for sbj_n in range(1):
    nnn = 3
    # Criação do objeto do sujeito que será analisado
    # sbj = IVBCICompetitionSubject(headset, {1: "l", 2: "r", 3: "f", 4: "t"}, f"A0{nnn}", timing_configs)
    # sbj.set_fif_files()       # Converte os arquivos de gravação originais no padrão .fif e separa entre test e train
    # sbj.save_object()         # Salva a instância criada do objeto Subject com todas as configurações dadas
    # sbj.generate_epochs()     # Guarda todas as epocas contidas nas gravações em arquivos separados por classe

    print("Carregando o sujeito")
    sbj = Subject.load_from_foldername("A03")

    print("Criando o classificador")
    classifier = OneVsOneLinearSVM(sbj)
    # classifier = OneVsOneLinearSVM.load_from_subjectname("A03")

    print("Salvando o classificador")
    classifier.save_classifier()

    print("Realizando testes sobre o classificador criado")
    res = classifier.run_testing_classifier()

    print(res)

