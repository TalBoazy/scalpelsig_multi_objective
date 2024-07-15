import os

PROJECT_DIR = "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective"
EXP_DIR = os.path.join(PROJECT_DIR, "icgc_exp")
WINDOWS_COUNT_DIR = os.path.join(EXP_DIR, "numpy_window_counts_matrices")
PATIENTS_INDEX_DIR = os.path.join(EXP_DIR, "patient_names")
WINDOWS_INDEX_DIR = os.path.join(EXP_DIR, "window_names")
MUTATIONS_INDEX_DIR = os.path.join(EXP_DIR, "mutation_names")
PROJECTION_SCORES_DIR = os.path.join(EXP_DIR, "projection_scores")
PROJECTION_TEMP_DIR = os.path.join(PROJECTION_SCORES_DIR, "temp")
TRAIN_TEST_DIR = os.path.join(EXP_DIR, "train_test")
NNLS_SCORES_DIR = os.path.join(EXP_DIR, "nnls_scores")
MMM_SCORES_DIR = os.path.join(EXP_DIR, "mmm_scores")
NMF_SCORES_DIR = os.path.join(EXP_DIR, "nmf_scores")
SELECTED_WINDOWS = os.path.join(EXP_DIR, "selected_windows")
PANELS = os.path.join(EXP_DIR, "sampled_panels")
RESULTS = os.path.join(EXP_DIR, "results")