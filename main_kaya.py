from classes.subjects import *
from classes.classifications import OneVsOneLinearSVM, OneVsAllLinearSVM, ConvolutionalClassifier
from classes.data_configuration import SubjectTimingConfigs
from classes.data_configuration import Headset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

sns.set(style="ticks")


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

resultados = {}

for classifier_method in [OneVsOneLinearSVM, OneVsAllLinearSVM, ConvolutionalClassifier]:
    resultados[classifier_method.__name__] = {}
    for sb_name in sbjs:
        # sbj = KayaDatasetSubject(headset, classes, sb_name, timing_configs)
        # sbj.set_fif_files()    # Converte os arquivos de gravação originais no padrão .fif e separa entre test e train
        # sbj.save_object()      # Salva a instância criada do objeto Subject com todas as configurações dadas
        # sbj.generate_epochs()  # Guarda todas as epocas contidas nas gravações em arquivos separados por classe

        print(f"Carregando o sujeito {sb_name}")
        sbj = Subject.load_from_foldername(sb_name)

        # print("Criando o classificador")
        # classifier = OneVsOneLinearSVM(sbj)
        # print("Salvando o classificador")
        # classifier_method.save_classifier()

        print("Carregando o classificador")
        classifier = classifier_method.load_from_subjectname(sb_name)

        print("Realizando testes sobre o classificador criado")
        res = classifier.run_testing_classifier()

        e_classes = list(classes.values())
        e_dict = classes

        y_test = res["real_classes"]
        y_prediction_final = res["predicted_classes"]

        acertos = np.array(y_test) == np.array(y_prediction_final)

        confusion_df = pd.DataFrame(
            np.zeros([len(e_classes), len(e_classes)]),
            index=e_classes, columns=e_classes
        )

        for i_cnt, i in enumerate(y_prediction_final):
            confusion_df.loc[e_dict[y_test[i_cnt]], e_dict[y_prediction_final[i_cnt]]] += 1

        confusion_df = confusion_df.rename(
            columns={"lh": "Mão Esquerda", "rh": "Mão Direita", "ll": "Pé Esquerdo", "rl": "Pé Direito",
                     "t": "Língua", "n": "relaxamento"},
            index={"lh": "Mão Esquerda", "rh": "Mão Direita", "ll": "Pé Esquerdo", "rl": "Pé Direito",
                   "t": "Língua", "n": "relaxamento"}
        )

        confusion = confusion_df.to_numpy()
        pe = np.trace(confusion) / np.sum(confusion)
        po = np.dot(np.sum(confusion, axis=0), np.sum(confusion, axis=1)) / (np.sum(confusion) ** 2)
        kappa = (pe - po) / (1 - po)

        confusion_percent = confusion
        ax = sns.heatmap(confusion_df, cmap="Blues", annot=confusion_percent, linewidths=1.5)
        plt.yticks(va="center")
        plt.xticks(va="center")
        plt.ylabel("Classe Real")
        plt.xlabel("Classe Predita")

        plt.title(f"{sb_name[1:]}")

        print(f"Taxa de acerto {sb_name}:", acertos.mean(), f"kappa: {kappa}")

        os.makedirs(f"C:/Users/victo/Desktop/{classifier_method.__name__}", exist_ok=True)

        ax.get_figure().savefig(
            f"C:/Users/victo/Desktop/{classifier_method.__name__}/{sb_name}_confusion.png", dpi=200
        )
        plt.cla()
        plt.clf()
        plt.close()

        resultados[classifier_method.__name__][sb_name] = {
            "kappa": kappa,
            "accuracy": acertos.mean()
        }

print(resultados)

# as requested in comment
exDict = {'exDict': resultados}

with open('C:/Users/victo/Desktop/file.txt', 'w') as file:
    file.write(json.dumps(exDict))  # use `json.loads` to do the reverse

          