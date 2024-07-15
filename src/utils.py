import numpy as np
import os
from scipy.sparse import csr_matrix, vstack
from src.raftery_at_al import WindowsDataManager
import pandas as pd
import json
from src.constants import *
from src.genetic_algorithm import GeneticAlgorithm
from src.kl_divergance_optimization import kl_divergence
from src.run_nnls import calculate_nnls
from data.data_utils import get_mutational_signatures


def return_compressed_chromosomes(path=WINDOWS_COUNT_DIR):
    files = os.listdir(path)
    chrom_dict = {}
    for file in files:
        chrom = file[5:-4]
        chrom_dict[chrom] = np.load(os.path.join(path, file))
    return chrom_dict


def get_window_score_files(tested_score, score_dir):
    score_files = os.listdir(score_dir)
    files_subset = []
    for f in score_files:
        if tested_score in f:
            files_subset.append(f)
    return files_subset


def get_train_samples(it):
    return np.sort(np.load(os.path.join(TRAIN_TEST_DIR, "train_it" + str(it) + ".npy")))


def get_test_samples(it):
    return np.sort(np.load(os.path.join(TRAIN_TEST_DIR, "test_it" + str(it) + ".npy")))


def get_positive_and_negative_samples(tested_score, it, thresh=0.2):
    train_samples = get_train_samples(it)
    # sig = int(tested_score[3:])
    sig = tested_score
    real_exposures = np.load(os.path.join(EXP_DIR, "nnls_exposures.npy"))[train_samples, sig - 1]
    positive_samples = np.where(real_exposures >= thresh)
    negative_samples = np.where(real_exposures < thresh)
    return positive_samples, negative_samples



def get_signatures_frequencies(sigs, it):
    freqs = []
    tot = 0
    for i in range(len(sigs)):
        pos, neg = get_positive_and_negative_samples(sigs[i], it)
        freq = len(pos[0])/(len(pos[0])+len(neg[0]))
        freqs.append(freq)
        tot += freq
    final_freqs = [freqs[i]/tot for i in range(len(freqs))]
    return final_freqs



def numpy_data_transform(num_samples=-1):
    data = csr_matrix((0, 49152), dtype=np.int64)
    samples = get_train_samples(1)
    with open(os.path.join(EXP_DIR, "windows_index_dict.json"), 'r') as json_file:
        # Load the JSON data into a Python dictionary
        index_dict = json.load(json_file)
    chrom_dict = return_compressed_chromosomes(WINDOWS_COUNT_DIR)
    window_manager = WindowsDataManager()
    window_manager.update_vals(chrom_dict, index_dict, 1000, 569)
    index = 0
    for key in index_dict:
        if num_samples != -1:
            if index > num_samples:
                break
        chrom, batch_index = eval(key)
        batch = window_manager.get_batch(chrom, batch_index)[samples]
        flatten_batch = batch.transpose(1, 0, 2).reshape(batch.shape[1], -1)
        sparse_batch = csr_matrix(flatten_batch)
        data = vstack([data, sparse_batch])
        index += batch.shape[1]
    return data


def sample_data_panel(samples, windows, output_file, windows_dir=WINDOWS_COUNT_DIR):
    """
    :param windows: df with chrom name and index
    :return:
    """
    count_mat = np.zeros(shape=(len(samples), 96))
    chrom_windows = windows.groupby("chrom")
    for chrom, chrom_df in chrom_windows:

        data_file = os.path.join(WINDOWS_COUNT_DIR, "chrom" + chrom + ".npz")
        compressed_chrom_mats = np.load(os.path.join(windows_dir, data_file))
        n_chunks = len(compressed_chrom_mats.files)
        index = 0
        for j in range(n_chunks):
            chrom_mat = compressed_chrom_mats[f'my_array{j}'][samples]
            lower_bound = index
            upper_bound = lower_bound + chrom_mat.shape[1]
            relevant_windows = chrom_df[
                (chrom_df['window_index'] >= lower_bound) & (chrom_df['window_index'] < upper_bound)]["window_index"]
            count_mat += chrom_mat[:, relevant_windows - index, :].sum(axis=1)
            index = upper_bound
    np.save(os.path.join(PANELS, output_file), count_mat)


def get_real_exposures(samples, sigs):
    real_exposures = np.load(os.path.join(EXP_DIR, "nnls_exposures.npy"))
    return real_exposures[samples][:, sigs]


def create_window_manager_object(json_file_path, it, train=False, debug_n = None):
    with open(json_file_path, 'r') as json_file:
        # Load the JSON data into a Python dictionary
        index_dict = json.load(json_file)
    chrom_dict = return_compressed_chromosomes(WINDOWS_COUNT_DIR)
    window_manager = WindowsDataManager(debug_n)
    if train:
        samples = get_train_samples(it)
    else:
        samples = get_test_samples(it)
    window_manager.update_vals(chrom_dict, index_dict, 1000, samples.shape[0], samples)
    return window_manager


def collect_count_mat_from_windows(window_manager, windows_df):
    count_mat = window_manager.get_entries_(windows_df["indices"], windows_df["chrom"], windows_df["batch"])
    return count_mat


def o(panel, real_pi, e):
    out, _ = calculate_nnls(panel, e)
    return kl_divergence(real_pi, out)


def write_list_to_file(file_path, data_list):
    """
    Writes a list to a file.

    Parameters:
    file_path (str): The path to the file where the list will be written.
    data_list (list): The list of data to write to the file.
    """
    try:
        with open(file_path, 'w') as file:
            for item in data_list:
                file.write(f"{item}\n")
        print(f"Successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


if __name__ == "__main__":
    # todo add to a function and also debug the counts, and also do clustering analysis
    window_manager = create_window_manager_object(os.path.join(EXP_DIR, "windows_index_dict.json"), 1, train=True, debug_n=2)
    pi = np.load("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\nnls_exposures.npy")
    pi = pi[window_manager.samples]
    g = GeneticAlgorithm(o, max_iter=3)
    e = get_mutational_signatures([0, 1, 2, 3, 4])
    best_chrom = g.fit(window_manager, pi, e)
    write_list_to_file("best_chrom.txt", best_chrom)
    df = pd.read_csv(os.path.join(SELECTED_WINDOWS, "dense_exp.csv"))
    count_mat = collect_count_mat_from_windows(window_manager, df)
    np.save(os.path.join(PANELS, "dense_exp.npy"),count_mat)
    alphas = [0.0001, 1e-5, 1e-10, 1e-20, 1e-30]
    for a in alphas:
        df = pd.read_csv(
            "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\selected_windows\\mix_for_windows_exp_" +'{:.0e}'.format(a) + ".csv")
        count_mat = collect_count_mat_from_windows(window_manager, df)
        np.save(os.path.join(PANELS, "mix_for_windows_exp_" + str(a) + ".npy"), count_mat)

    # count_mat = window_manager.get_entries_(df["indices"], df["chrom"], df["batch"])
    window_file = pd.read_csv(os.path.join(SELECTED_WINDOWS, "NNLS_binary_model_sigs_1_2_3_5_6_it1.csv"))
    numpy_mat_train = os.path.join(PANELS, "NNLS_binary_model_sigs_1_2_3_5_6_it1_train.npy")
    numpy_mat_test = os.path.join(PANELS, "NNLS_binary_model_sigs_1_2_3_5_6_it1_test.npy")
    # windows_files = [pd.read_csv(
    #     "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\src\\generation_0_score_1.134.csv"),
    #                  pd.read_csv(
    #                      "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\src\\generation_3_score_1.115.csv"),
    #                  pd.read_csv(
    #                      "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\src\\generation_6_score_1.087.csv"),
    #                  pd.read_csv(
    #                      "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\src\\generation_9_score_1.047.csv"),
    #                  pd.read_csv(
    #                      "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\src\\generation_19_score_0.973.csv")]
    #
    # numpy_paths = [os.path.join(PANELS, "generation_0_score_1.134_train.npy"),
    #                os.path.join(PANELS, "generation_3_score_1.115_train.npy"),
    #                os.path.join(PANELS, "generation_6_score_1.087_train.npy"),
    #                os.path.join(PANELS, "generation_9_score_1.047_train.npy"),
    #                os.path.join(PANELS, "generation_19_score_0.973_train.npy")]
    # sample_data_panel(pd.read_csv(
    #     "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\src\\generation_15_score_0.495.csv"),
    #                   1, os.path.join(PANELS, "generation_15_score_0.495.npy"))
    train_samples = get_train_samples(1)
    test_samples = get_test_samples(1)
    sample_data_panel(train_samples, window_file, 1, numpy_mat_train)
    sample_data_panel(test_samples, window_file, 1, numpy_mat_test)
    # for i in range(len(windows_files)):
    #     sample_data_panel(train_samples, windows_files[i], 1, numpy_paths[i], "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\numpy_window_counts_matrices_clustered_it1")
    # windows = pd.read_csv(
    #    "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\selected_windows\\NNLS_binary_model_sig1sig5_it1_500.csv")
    # sample_data_panel(windows,1, "NNLS_binary_model_sig1sig5_it1_500.npy")
    # prefix = "NMF_sig1sig5_it1_"
    # suffix = ["_5_clusters.csv","_25_clusters.csv", "_40_clusters.csv", "_50_clusters.csv"]
    # for label in suffix:
    #     windows = pd.read_csv(os.path.join(SELECTED_WINDOWS, prefix+label))
    #     sample_data_panel(windows, 1, "nmf_sig1sig5"+label[:-4]+".npy")
    # windows = pd.read_csv(
    #     "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\selected_windows\\nmf_greedy_aggregation_sig1sig5.csv")
    # sample_data_panel(windows, 1, "mf_greedy_aggregation_sig1sig5.npy")
    # windows = pd.read_csv("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\selected_windows\\NMF_sig1sig5_it1_kl_measured.csv")
    # sample_data_panel(windows, 1, "NMF_sig1sig5_it1_count_mat_kl_measured.npy")
    # windows = pd.read_csv(
    #   "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\selected_windows\\NNLS_binary_model_sig1_it1.csv")
    # sample_data_panel(windows, 1, "NNLS_binary_model_sig1_it1_count_mat.npy")
