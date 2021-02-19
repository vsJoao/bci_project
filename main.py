from classes.subjects import IVBCICompetitionSubject
from classes.datasets import SubjectTimingConfigs
from classes.datasets import Headset
from classes.classifications import OneVsOneLinearSVM

# timing_configs = SubjectTimingConfigs(
#     trial_duration=7.5, sample_start=3.5, sample_end=6, ica_start=0, ica_end=7, epc_size=None, time_between=None
# )
# headset = Headset.from_headset_name("iv_bci_headset")
# sbj = IVBCICompetitionSubject(headset, {1: "l", 2: "r", 3: "f", 4: "t"}, "A01", timing_configs)
# sbj.set_fif_files()
# sbj.save_object()
# sbj.generate_epochs()
sbj = IVBCICompetitionSubject.load_from_foldername("A01")
# sbj.generate_fbcsp_one_vs_one()
# sbj.generate_subject_train_features_one_vs_one()
# sbj.generate_subject_test_features_one_vs_one()
# sbj.generate_one_vs_one_svmlinear_classifier()
# classifier = OneVsOneLinearSVM.load_from_subjectname("A01")
sbj.run_testing_one_vs_one_svmlinear_classifier()
