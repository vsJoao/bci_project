from classes.subjects import *
from classes.classifications import OneVsOneLinearSVM, OneVsAllLinearSVM, ConvolutionalClassifier
from classes.data_configuration import SubjectTimingConfigs
from classes.data_configuration import Headset
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Criação do objeto de temporização do sinal
# timing_configs = SubjectTimingConfigs(
#     trial_duration=7.5, sample_start=3.5, sample_end=6, ica_start=0, ica_end=7, epc_size=None, time_between=None
# )
timing_configs = SubjectTimingConfigs(
    trial_duration=7.5, sample_start=3.5, sample_end=6, ica_start=0, ica_end=7, epc_size=None, time_between=None
)

# Canais que são utilizados no dataset
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
            'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
            'EOG1', 'EOG2', 'EOG3']

# Criação do objeto do headset que foi feita a gravação
headset = Headset(ch_names=ch_names, max_signal=100, sfreq=250, name="iv_bci_headset")
headset.save_headset()

sbjs = ["A01", "A02", "A03", "A05", "A06", "A07", "A08", "A09"]
classes = {1: "l", 2: "r", 3: "f", 4: "t"}
resultados = {}

for classifier_method in [OneVsAllLinearSVM]:
    resultados[classifier_method.__name__] = {}
    for sbj_n in sbjs:
        nnn = sbj_n
        # Criação do objeto do sujeito que será analisado
        # sbj = IVBCICompetitionSubject(headset, {1: "l", 2: "r", 3: "f", 4: "t"}, nnn, timing_configs)
        # sbj.set_fif_files()    # Converte os arquivos de gravação originais no padrão .fif e separa entre test e train
        # sbj.save_object()      # Salva a instância criada do objeto Subject com todas as configurações dadas
        # sbj.generate_epochs()  # Guarda todas as epocas contidas nas gravações em arquivos separados por classe

        print("Carregando o sujeito")
        sbj = Subject.load_from_foldername(nnn)

        print("Criando o classificador")
        classifier = classifier_method(sbj)

        # print("Salvando o classificador")
        # classifier.save_classifier()

        # print("Carregando o classificador")
        # classifier = OneVsOneLinearSVM.load_from_subjectname(nnn)

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
            columns={"l": "Mão Esquerda", "r": "Mão Direita", "f": "Pés", "t": "Lìngua"},
            index={"l": "Mão Esquerda", "r": "Mão Direita", "f": "Pés", "t": "Lìngua"}
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

        plt.title(f"{nnn[1:]}")

        print(f"Taxa de acerto {nnn}:", acertos.mean(), f"kappa: {kappa}")

        os.makedirs(f"C:/Users/victo/Desktop/{classifier_method.__name__}", exist_ok=True)

        ax.get_figure().savefig(
            f"C:/Users/victo/Desktop/{classifier_method.__name__}/{nnn}_confusion.png", dpi=200
        )
        plt.cla()
        plt.clf()
        plt.close()

        resultados[classifier_method.__name__][nnn] = {
            "kappa": kappa,
            "accuracy": acertos.mean(),
            "real": res["real_classes"],
            "predicted": res["predicted_classes"]
        }

print(resultados)

# as requested in comment
exDict = {'exDict': resultados}

with open('C:/Users/victo/Desktop/file_bcicompetition.txt', 'w') as file:
    file.write(json.dumps(exDict))  # use `json.loads` to do the reverse


