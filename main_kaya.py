from classes.subjects import *
from classes.classifications import OneVsOneLinearSVM, OneVsAllLinearSVM, ConvolutionalClassifier
from classes.data_configuration import SubjectTimingConfigs
from classes.data_configuration import Headset


sbjs = {
    'K01': 'A',
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
    'K12': 'M'
}

# Criação do objeto de temporização do sinal
timing_configs = SubjectTimingConfigs(
    trial_duration=1, sample_start=0.1, sample_end=0.9, ica_start=0, ica_end=1, epc_size=None, time_between=None
)

# Canais que são utilizados no dataset
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8',
            'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'F5']

# Classes utilizadas no dataset
classes = {
    1: 'lh',    # left hand
    2: 'rh',    # right hand
    3: 'n',     # neutral
    4: 'll',    # left leg
    5: 't',     # tongue
    6: 'rl',    # right leg
    # 91: 'b',    # beginning
    # 92: 'e',    # experiment end
    # 99: 'r'     # inicial relaxing
}

# Criação do objeto do headset que foi feita a gravação
headset = Headset(ch_names=ch_names, sfreq=200, name="kaya_ds_headset", max_signal=100)
headset.save_headset()

for sb in sbjs:
    # sbj = KayaDatasetSubject(headset, classes, sb, timing_configs)
    # sbj.set_fif_files()       # Converte os arquivos de gravação originais no padrão .fif e separa entre test e train
    # sbj.save_object()         # Salva a instância criada do objeto Subject com todas as configurações dadas
    # sbj.generate_epochs()     # Guarda todas as epocas contidas nas gravações em arquivos separados por classe

    print(f"Carregando o sujeito {sb}")
    sbj = Subject.load_from_foldername(sb)
    print("Carregando Modelo Classificador")
    classifier = OneVsOneLinearSVM.load_from_subjectname(sb)

    # print("Criando o classificador")
    # classifier = OneVsOneLinearSVM(sbj)
    # print("Salvando o classificador")
    # classifier.save_classifier()

    # Carregando o classificador
    # classifier = ConvolutionalClassifier(sbj)
    # classifier.save_classifier()

    print("Realizando testes sobre o classificador criado")
    res = classifier.run_testing_classifier()
    print(res)

