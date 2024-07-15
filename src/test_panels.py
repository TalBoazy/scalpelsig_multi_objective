import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from src.run_nnls import calculate_nnls
from src.utils import get_test_samples, get_real_exposures, get_train_samples
from data.data_utils import get_mutational_signatures
import os
from src.constants import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import json


def calculate_aupr(panel, sigs, it, sigs_indices):
    non_zero_rows = np.where(panel.any(axis=1))[0]
    panel = panel[non_zero_rows]
    exposures, _ = calculate_nnls(panel, sigs)
    exposures = (exposures.T / exposures.sum(axis=1)).T
    test_samples = get_train_samples(it)
    true_exposures = get_real_exposures(test_samples, sigs_indices)[non_zero_rows]
    binary_true_exposures = true_exposures >= 0.1
    binary_exposures = exposures >=0.1
    # calc aupr for every sig
    auprs = []
    for i in range(len(sigs_indices)):
        precision, recall, _ = precision_recall_curve(binary_true_exposures[:,i], binary_exposures[:,sigs_indices[i]])
        aupr = auc(recall, precision)
        auprs.append(aupr)
    print(np.array(auprs).mean())
    return auprs



def calculate_aupr_multiclass(panel, sigs, it, sigs_indices, cluster_dict):
    non_zero_rows = np.where(panel.any(axis=1))[0]
    panel = panel[non_zero_rows]
    exposures, _ = calculate_nnls(panel, sigs)
    exposures = (exposures.T / exposures.sum(axis=1)).T
    test_samples = get_train_samples(it)
    true_exposures = get_real_exposures(test_samples, sigs_indices)[non_zero_rows]
    binary_true_exposures = (true_exposures >= 0.1).view(np.int8)
    binary_exposures = (exposures >= 0.1).view(np.int8)
    clusters = np.array([list(eval(key)) for key in cluster_dict.keys()])
    auprs = []
    for i in range(clusters.shape[0]):
        real_classification = np.all(binary_true_exposures == clusters[i], axis=1).astype(int)
        model_classification = np.all(binary_exposures == clusters[i], axis=1).astype(int)
        precision, recall, _ = precision_recall_curve(real_classification, model_classification)
        aupr = auc(recall, precision)
        auprs.append(aupr)
    print(np.array(auprs).mean())
    # calc aupr for every sig


def sanity(panel, sigs, it, sigs_indices):
    non_zero_rows = np.where(panel.any(axis=1))[0]
    panel = panel[non_zero_rows]
    exposures, _ = calculate_nnls(panel, sigs)
    exposures = (exposures.T / exposures.sum(axis=1)).T
    test_samples = get_test_samples(it)
    true_exposures = get_real_exposures(test_samples, sigs_indices)
    true_exposures = true_exposures[non_zero_rows]
    binary_exposures = (exposures > 0.2).astype(int)
    binary_true_exposures = (true_exposures > 0.2).astype(int)
    scores = 0
    for i in range(len(sigs_indices)):
        score = np.abs(binary_exposures[:, sigs_indices[i]].flatten()- binary_true_exposures[:,i].flatten()).sum()
        scores += score
    print(scores / len(sigs_indices))

def spearman_test(panel, sigs, it, sigs_indices):
    non_zero_rows = np.where(panel.any(axis=1))[0]
    panel = panel[non_zero_rows]
    exposures, _ = calculate_nnls(panel, sigs)
    exposures = (exposures.T / exposures.sum(axis=1)).T
    test_samples = get_test_samples(it)
    true_exposures = get_real_exposures(test_samples, sigs_indices)
    true_exposures = true_exposures[non_zero_rows]
    rho, p_value = spearmanr(exposures[:, sigs_indices].flatten(), true_exposures.flatten())
    rho_tot = 0
    for i in range(len(sigs_indices)):
        rho, p_value = spearmanr(exposures[:, sigs_indices[i]].flatten(), true_exposures[:,i].flatten())
        rho_tot+=rho
    print("average")
    print(rho_tot/len(sigs_indices))
    return rho, p_value


def run_exps_and_save(names, data_path, sigs, it, sig_indices, output_name):
    tuples = []
    for i in range(len(names)):
        data = np.load(data_path[i])
        corr, p_val = spearman_test(data, sigs, it, sig_indices[i])
        tuples.append((names[i], corr, p_val))
    result_df = pd.DataFrame(tuples, columns=['exp', 'spearman correlation', 'p value'])
    result_df.to_csv(os.path.join(RESULTS, output_name))


numpy_paths = [os.path.join(PANELS, "mix_for_windows_exp_1e-30.npy"),
                   os.path.join(PANELS, "mix_for_windows_exp_1e-20.npy"),
                   os.path.join(PANELS, "mix_for_windows_exp_1e-10.npy"),
                   os.path.join(PANELS, "mix_for_windows_exp_1e-05.npy"),
                   os.path.join(PANELS, "mix_for_windows_exp_0.0001.npy"),
                    os.path.join(PANELS, "NNLS_binary_model_sigs_1_2_3_5_6_it1_test.npy",
                                 os.path.join(PANELS, "dense_exp.npy"))
               ]

sigs = get_mutational_signatures(np.array([1, 2, 3, 5, 6]) - 1)
with open(os.path.join(EXP_DIR, "clusters_dict_basic_it1.json"), "r") as f:
    cluster_dict = json.load(f)
for mat in numpy_paths:
    m = np.load(mat)
    #sanity(m,sigs,1,[0,1,2,3,4])
    #calculate_aupr_multiclass(m, sigs, 1, [0,1,2,3,4], cluster_dict)
    print(spearman_test(m,sigs,1,[0,1,2,3,4]))


# data = np.load(os.path.join(PANELS, "NNLS_binary_model_sig1sig5_it1_500.npy"))
# sigs = get_mutational_signatures(np.array([1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]) - 1)
# calculate_aupr(data, sigs, 1, [0, 3])
#
# data = np.load(os.path.join(PANELS, "NNLS_binary_model_sig2_it1.npy"))
# sigs = get_mutational_signatures(np.array([1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]) - 1)
# corr, p_val = spearman_test(data, sigs, 1, [1])
#
# names = ["msk_impact", "nmf_5_clusters", "nmp_25_clusters", "nmf_40_clusters", "nmf_50_clusters"]
# data_path = [os.path.join(EXP_DIR, "icgc_brca_msk_impact_downsample.npy"),
#              os.path.join(PANELS, "nmf_sig1sig5_5_clusters.npy"),
#              os.path.join(PANELS, "nmf_sig1sig5_25_clusters.npy"), os.path.join(PANELS, "nmf_sig1sig5_40_clusters.npy"),
#              os.path.join(PANELS, "nmf_sig1sig5_50_clusters.npy")]
# sig_indices = [[0, 3], [0, 3], [0, 3], [0, 3], [0, 3]]
# output_name = "aggregated_nmf_results.csv"
#
# run_exps_and_save(names, data_path, sigs, 1, sig_indices, output_name)
# exp_sig1_panel = np.load("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\sampled_panels\\NNLS_binary_model_sig1_it1_count_mat.npy")
# res_sig1 = spearman_test(exp_sig1_panel, sigs,1,[0])
# scalplelsig_panel = np.load("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\sampled_panels\\NNLS_binary_model_sig1sig5_it1_count_mat.npy")
# res_scalplelsig_multi=spearman_test(scalplelsig_panel, sigs,1,[0,3])
# nnls_panel = np.load("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\sampled_panels\\NNLS_multi_objective_sig1sig5_it1_count_mat.npy")
# res_nnls=spearman_test(nnls_panel, sigs,1,[0,3])
# nmf_panel = np.load("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\sampled_panels\\NMF_sig1sig5_it1_count_mat.npy")
# res_nmf = spearman_test(nmf_panel, sigs,1,[0,3])
