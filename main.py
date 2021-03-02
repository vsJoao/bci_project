from classes.subjects import IVBCICompetitionSubject
from classes.classifications import OneVsOneLinearSVM
from classes.data_configuration import SubjectTimingConfigs
from classes.data_configuration import Headset
from classes.classifications import OneVsOneLinearSVM

timing_configs = SubjectTimingConfigs(
    trial_duration=7.5, sample_start=3.5, sample_end=6, ica_start=0, ica_end=7, epc_size=None, time_between=None
)
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
            'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
            'EOG1', 'EOG2', 'EOG3']

headset = Headset(ch_names=ch_names, sfreq=250, name="iv_bci_headset")
headset.save_headset()
sbj = IVBCICompetitionSubject(headset, {1: "l", 2: "r", 3: "f", 4: "t"}, "A08", timing_configs)
sbj.set_fif_files()
sbj.save_object()
sbj.generate_epochs()

# print("Carregando o sujeito")
# sbj = IVBCICompetitionSubject.load_from_foldername("A01")
# headset = Headset.from_headset_name("iv_bci_headset")

print("Criando o classificador")
classifier = OneVsOneLinearSVM(sbj)
print("Salvando o classificador")
classifier.save_classifier()
print("Realizando testes sobre o classificador criado")
res = classifier.run_testing_classifier()
print(res)
