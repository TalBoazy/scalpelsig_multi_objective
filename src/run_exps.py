import sys
sys.path.append("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective")
import os.path
from constants import *
from data.data_utils import get_mutational_signatures
from src.utils import get_real_exposures, get_train_samples, numpy_data_transform, return_compressed_chromosomes
from src.genetic_algorithm_basic_scheme import BasicGeneticAlgorithm
from src.genetic_algorithm_basic_scheme import WindowsDataManager
from src.raftery_at_al import run_exp
from src.MIX.Mix import Mix
from src.mmm import MMM
import json
import numpy as np
import random
from itertools import product
import pandas as pd
from src.feature_importance_methods import shuffle
from src.run_nnls import calculate_nnls
from src.Mix_for_windows import MixWindows
import json


def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


def save_model(path, model):
    res_dict = {"exposures": model.pi.tolist(), "clusters": model.w.tolist(), "signatures": model.e.tolist(),
                "k": model.num_topics, "l": model.num_clusters}
    with open(path, "w+") as fp:
        json.dump(res_dict, fp)


def count_groups(sigs):
    cluster_dict = {}
    train_samples = get_train_samples(1)
    samples = train_samples
    exposures = get_real_exposures(samples, sigs)
    binary_exposures = exposures >= 0.1
    binary_classes = list(product([0, 1], repeat=exposures.shape[1]))
    for group in binary_classes:
        cluster_group = np.where(np.all(binary_exposures == group, axis=1))[0]
        if cluster_group.shape[0]:
            cluster_dict[str(group)] = train_samples[cluster_group].tolist()
        print(group)
    return cluster_dict


def run_exp_feature_selection_for_mmm():
    with open(os.path.join(EXP_DIR, "windows_index_dict.json"), 'r') as json_file:
        # Load the JSON data into a Python dictionary
        index_dict = json.load(json_file)
    chrom_dict = return_compressed_chromosomes(WINDOWS_COUNT_DIR)
    samples = get_train_samples(1)
    exposures = get_real_exposures(samples, [0, 1, 2, 3, 4])
    exposures = (exposures.T / exposures.sum(axis=1)).T
    sigs = get_mutational_signatures([0, 1, 2, 3, 4])
    trained_model = MMM(init_params={"e": sigs, "pi": exposures}, k=5)
    untrained_model = Mix(6, 5, num_words=96, init_params={"e": sigs})
    window_manager = WindowsDataManager()
    window_manager.update_vals(chrom_dict, index_dict, 1000, 569)
    panel = run_exp(250, trained_model, untrained_model, window_manager, samples)
    df = pd.DataFrame(panel, columns=['chrom', 'batch', 'index'])
    df.to_csv(SELECTED_WINDOWS, "raftery_at_al.csv")


def jensen_shannon_divergence(p, q, epsilon=1e-10):
    """
    Calculate the Jensen-Shannon Divergence between two probability distributions.

    Parameters:
        p (numpy.ndarray): First probability distribution.
        q (numpy.ndarray): Second probability distribution.

    Returns:
        float: Jensen-Shannon Divergence between the distributions.
    """
    p += epsilon
    q += epsilon
    # Compute the average distribution
    m = (p + q) / 2

    # Compute the Jensen-Shannon Divergence
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    js_divergence = (kl_pm + kl_qm) / 2

    return js_divergence


def basic_distance_matrix(mat1, mat2):
    distances = np.zeros(shape=(mat1.shape[0]))
    for i in range(distances.shape[0]):
        distances[i] = jensen_shannon_divergence(mat1[i], mat2[i])
    return distances.mean()


def run_shuffle_method(d):
    with open(os.path.join(EXP_DIR, "windows_index_dict.json"), 'r') as json_file:
        # Load the JSON data into a Python dictionary
        index_dict = json.load(json_file)
    samples = get_train_samples(1)
    count_mat = pd.read_csv(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\count_mat.tsv")
    count_mat = count_mat.iloc[:, 1:].values[samples]
    real_exposures = get_real_exposures(samples, [0, 1, 2, 3, 4])
    real_exposures = (real_exposures.T / real_exposures.sum(axis=1)).T
    sigs = get_mutational_signatures([0, 1, 2, 3, 4])
    chrom_dict = return_compressed_chromosomes(WINDOWS_COUNT_DIR)
    window_manager = WindowsDataManager()
    window_manager.update_vals(chrom_dict, index_dict, 1000, 569)
    scores = []
    for key in index_dict:
        chrom, batch_index = eval(key)
        batch = window_manager.get_batch(chrom, batch_index)[samples]
        shuffled_batch = shuffle(batch)
        batch_scores = []
        for i in range(batch.shape[1]):
            shuffled_mat = count_mat - batch[:, i, :] + shuffled_batch[:, i, :]
            shuffled_exposures = calculate_nnls(shuffled_mat, sigs)[0]
            batch_scores.append(d(real_exposures, shuffled_exposures))
        batch_df = pd.DataFrame({"score": batch_scores, "index": range(index_dict[key][0], index_dict[key][1])})
        batch_df["chrom"] = chrom
        batch_df["batch_index"] = batch_index
        scores.append(batch_df)
    final_scores = pd.concat(scores)
    selected_scores = final_scores.sort_values(by='score', ascending=False).head(250)
    selected_scores.to_csv(os.path.join(SELECTED_WINDOWS, "shuffle_feature_importance"))
    return selected_scores


def cluster(trained_model, data_manager):
    clusters, topics, probabilites, counts, indices, chroms, batch = trained_model.predict(data_manager)
    df= pd.DataFrame({"probabilities":probabilites, "clusters": clusters, "counts":counts,
                      "indices":indices, "chrom":chroms, "batch":batch})
    df.to_csv(os.path.join(RESULTS, "mix_for_windows_exp.csv"), index=False)
    np.save(os.path.join(RESULTS, "windows_clusters.npy"), clusters)
    np.save(os.path.join(RESULTS, "windows_clusters_probabilities.npy"), probabilites)
    np.save(os.path.join(RESULTS, "clusters_topics.npy"), topics)
    np.save(os.path.join(RESULTS, "clusters_sum.npy"), counts)


def find_top_dense_windows(data_manager,select):
    selected = np.full(shape=(select,), fill_value=-np.inf)
    df = pd.DataFrame({"score":selected})
    df["indices"] = None
    df["chrom"] = None
    df["batch"] = None
    prev_chrom = -1
    # working on each set of windows separately
    for key in data_manager.window_index:
        (chrom, batch_index) = eval(key)
        if chrom != prev_chrom:
            print("currently processing chrom " + chrom)
            prev_chrom = chrom
        data = data_manager.get_batch(chrom, batch_index)[data_manager.samples].astype('int')
        scores = data.sum(axis=(0,2))
        new_df = pd.DataFrame({"score":scores, "indices":np.arange(data.shape[1])})
        new_df["chrom"] = chrom
        new_df["batch"] = batch_index
        combined_df = pd.concat([df,new_df])
        df = combined_df.sort_values(by="score",ascending=False).head(select)
        df.reset_index(inplace=True, drop=True)
    return df


def load_trained_model(path):
    with open(path, 'r') as json_file:
        # Load the JSON data into a Python dictionary
        model_dict = json.load(json_file)
    return MixWindows(5,5,init_params={"e":np.array(model_dict["signatures"]), "pi":np.array(model_dict["exposures"]),
                                   "w":np.array(model_dict["clusters"])})


if __name__ == "__main__":
    with open(os.path.join(EXP_DIR, "windows_index_dict.json"), 'r') as json_file:
        # Load the JSON data into a Python dictionary
        index_dict = json.load(json_file)
    chrom_dict = return_compressed_chromosomes(WINDOWS_COUNT_DIR)
    window_manager = WindowsDataManager()
    samples = get_train_samples(1)
    window_manager.update_vals(chrom_dict, index_dict, 1000, 512, samples)
    df = find_top_dense_windows(window_manager, 250)
    df.to_csv(os.path.join(SELECTED_WINDOWS, "dense_exp.csv"), index=False)
    df = pd.read_csv("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\selected_windows\\mix_for_windows_exp_0.0001.csv")
    window_manager.get_entries_(df["indices"], df["chrom"], df["batch"])
    e = get_mutational_signatures([0, 1, 2, 3, 4])
    mix_w = load_trained_model("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\results\\trained_mix_for_windows.json")
    #mix_w = MixWindows(num_clusters=5, num_topics=5, init_params={"e": e}, max_iter=1)
    mix_w.num_samples = 512
    cluster(mix_w, window_manager)
    # run_shuffle_method(basic_distance_matrix)
    # run_exp_feature_selection_for_mmm()
    # cluster_dict = count_groups([0,1,2,3,4])

    # with open(os.path.join(EXP_DIR,"clusters_dict_basic_it1.json"),"r") as f:
    #     cluster_dict = json.load(f)
    # n_generations = 20
    # n_children = 3
    # sigs = get_mutational_signatures([0, 1, 2, 3, 4])
    # scoring_func = lambda batch: calculate_distance_by_cluster(batch, sigs, cluster_dict)
    # # scoring_func = lambda batch: calculate_nnls_per_window(batch, sigs)
    # n_selected = 250
    # n_samples = 15
    # n_parents = 0.7
    # with open(os.path.join(EXP_DIR, "windows_index_dict.json"), 'r') as json_file:
    #     # Load the JSON data into a Python dictionary
    #     index_dict = json.load(json_file)
    # chrom_dict = return_compressed_chromosomes("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\numpy_window_counts_matrices_clustered_it1")
    # initial_partition = list(range(282000))
    # genetic_method = BasicGeneticAlgorithm(n_generations, n_children, 0, 0, scoring_func,
    #                                        n_selected,n_samples,n_parents)
    # res = genetic_method.fit(chrom_dict,index_dict,1000,n_samples,initial_partition)
    # res.to_csv(os.path.join(SELECTED_WINDOWS, "genetic_alg_exp_new_loss.csv"))
    # window_manager = WindowsDataManager()
    # window_manager.update_vals(chrom_dict,index_dict,1000,569)
    # indices = random.sample(initial_partition, 500)
    # scores = [0]*500
    # df = window_manager.transform_index_to_index_by_chrom(scores, indices)
    # df.to_csv(os.path.join(SELECTED_WINDOWS, "random_exp.csv"))
