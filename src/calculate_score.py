import numpy as np
import os
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from joblib import Parallel, delayed
from scipy.special import logsumexp
from data.data_utils import get_mutational_signatures
from src.run_nnls import calculate_nnls
from src.utils import get_train_samples, get_positive_and_negative_samples, get_real_exposures, create_window_manager_object
from src.constants import *
from src.mmm import MMM


def calculate_distance_from_real_exposures(batch, sigs, sigs_indices, it, thresh=0.1):
    train_samples = get_train_samples(it)
    batch = batch[train_samples]
    scores = np.zeros(shape=(batch.shape[1]))
    real_exposures = get_real_exposures(train_samples, sigs_indices)
    binary_exposures = (real_exposures >= thresh).astype(int)
    for i in range(batch.shape[1]):
        predicted_panel_exposures, _ = calculate_nnls(batch[:, i, :], sigs)
        binary_predicted_exposures = (predicted_panel_exposures>=thresh).astype(int)
        score_per_sample = np.linalg.norm(binary_exposures - binary_predicted_exposures, ord=2, axis=1)
        scores[i] = score_per_sample.mean()
    return scores


def calculate_distance_by_cluster(batch, sigs, cluster_dict):
    clusters = np.array([list(eval(key)) for key in cluster_dict.keys()])
    cluster_sizes = np.array([len(group) for group in cluster_dict.values()])
    scores = np.zeros(shape=(batch.shape[1],))
    for j in range(batch.shape[1]):
        predicted_exposures = calculate_nnls(batch[:,j,:],sigs)[0]+1e-30
        predicted_exposures = (predicted_exposures.T / predicted_exposures.sum(axis=1)).T
        scores[j] = (np.linalg.norm(predicted_exposures-clusters,ord=2,axis=1)*cluster_sizes).sum()/512
        if scores[j] != scores[j]:
            print("NONE")
    return scores



def mle_with_em_per_window(batch, sigs):
    mmm_model = MMM(k=sigs.shape[0], init_params={"e": sigs})
    ll = np.zeros(shape=(1, batch.shape[1]))
    for i in range(batch.shape[1]):
        window_data = batch[:, i, :]
        nonzero_rows = np.any(window_data != 0, axis=1)

        # Select rows that are not all zeros
        window_data = window_data[nonzero_rows]
        if not window_data.shape[0]:
            continue
        mmm_model.fit(window_data)
        pi = mmm_model.pi
        ll[i] = mmm_model.log_likelihood(batch, pi)
    return ll


def kl_divergence(X, E, pi):
    P, Q = X, E.dot(pi)
    return (P * np.log(P / Q) - P + Q).sum()


def calculate_the_grad(X, E, pi):
    return -E.T.dot(X / E.dot(pi)) + E.T.dot(np.ones(shape=X.shape))


def gradient_based_optimization_with_nmf(X, E, pi, tau, epsilon):
    if np.any(X == 0):
        X = X + 1e-30
    f_old = np.inf
    f_new = kl_divergence(X, E.T, pi)
    # grad = calculate_the_grad(X,E.T,pi)
    while np.abs(f_new - f_old) > epsilon:
        f_old = f_new
        # step_size = pi/E.dot(E.T).dot(pi)
        pi = pi * E.dot(X / E.T.dot(pi)) / np.ones(shape=pi.shape)
        f_new = kl_divergence(X, E.T, pi)
    return pi, f_new


def calculate_nmf_per_window(batch, sigs, final_score_func):
    mmm_model = MMM(k=sigs.shape[0], init_params={"e": sigs})
    log_likelihood_score = np.zeros(shape=(batch.shape[1]))
    for j in range(batch.shape[1]):
        if not batch[:, j, :].sum():
            ll = np.inf
            if not final_score_func:
                ll = -ll
        else:
            non_empty_window = (batch[:, j, :])[~np.all(batch[:, j, :] == 0, axis=1)]
            mutation_count_per_sample = non_empty_window.sum(axis=1)
            non_empty_window = (non_empty_window.T / mutation_count_per_sample).T
            ll = np.zeros(shape=(non_empty_window.shape[0]))
            for i in range(non_empty_window.shape[0]):
                pi = np.random.rand(2)
                pi = pi / pi.sum()
                pi, kl = gradient_based_optimization_with_nmf(non_empty_window[i], sigs, pi,
                                                              0.2,
                                                              1e-10)
                ll[i] = mmm_model.log_likelihood(non_empty_window[i][np.newaxis, :], pi[np.newaxis, :])
            if not final_score_func:
                ll = ll.mean()
            else:
                ll = kl
        log_likelihood_score[j] = ll
    return log_likelihood_score


def calculate_nnls_per_window(batch, sigs):
    # todo - select only train samples
    scores = np.zeros(shape=(batch.shape[1]))
    for j in range(batch.shape[1]):
        if not batch[:, j, :].sum():
            least_squares = np.inf
        else:
            non_empty_window = batch[:, j, :][~np.all(batch[:, j, :] == 0, axis=1)]
            mutation_count_per_sample = non_empty_window.sum(axis=1)
            non_empty_window = (non_empty_window.T / mutation_count_per_sample).T
            _, least_squares = calculate_nnls(non_empty_window, sigs)
        scores[j] = least_squares
    return scores


def batch_score_calculation_euclidean_norm(batch, sig, tested_score, it, alpha=0.5):
    positive, negative = get_positive_and_negative_samples(tested_score, it, thresh=0.05)
    scores = batch.dot(sig) / sig.dot(sig)
    scores = scores ** alpha
    return scores[positive].sum(axis=0) - scores[negative].sum(axis=0)


def write_the_final_score_matrix_to_rds(mat, patient_attrs, window_attrs, output_path):
    numpy2ri.activate()
    r = robjects.r
    r['source']('../data/R_scripts/create_rds_file.R')

    args = [mat, patient_attrs, window_attrs, output_path]
    rds_func = robjects.globalenv['create_rds_projection_score_lst']
    rds_func(*args)


def calculate_log_gradient(log_pi, A_k):
    return A_k - log_pi


def calculate_A_k(log_pi, log_e, log_x):
    # notice that since we maximize pi, there's no need to calculate Ejloge.sum since it's constant
    log_prob_mut_sig = log_e + log_pi
    log_prob_mut = logsumexp(log_prob_mut_sig, axis=1)
    log_prob_mut_per_sig = log_prob_mut_sig - log_prob_mut[:, np.newaxis]
    log_E_km = log_x[:, np.newaxis] + log_prob_mut_per_sig
    log_A_k = logsumexp(log_E_km, axis=0)
    return np.exp(log_A_k)


def log_likelihood(log_pi, A_k):
    return (A_k + log_pi).sum()


def gradient_ascent(f, df, alpha, parameters, epsilon, additional_args):
    theta = parameters

    while True:
        # Compute gradient
        gradient = df(theta, *additional_args)

        # Update parameters using gradient ascent
        theta_new = [theta_i + alpha * grad_i for theta_i, grad_i in zip(theta, gradient)]

        # Check convergence
        if abs(f(theta_new, *additional_args) - f(theta, *additional_args)) < epsilon:
            break

        # Update parameters
        theta = theta_new

    return theta


def calculate_clustered_matrix(batch, labels):
    labels_unique = np.unique(labels)
    n_clusters = labels_unique.shape[0]
    new_batch = np.zeros(shape=(n_clusters, batch.shape[1], batch.shape[2]))
    for i in range(n_clusters):
        clusters_members = batch[labels == labels_unique[i]]
        new_batch[i] = clusters_members.sum(axis=0)
    return new_batch


def calculate_projection_scores(sigs, sig_names, scoring_func, sigs_indices_iteration, out_dir, it,
                                max_window_in_a_batch_num=1000, labels=None):
    files = os.listdir(WINDOWS_COUNT_DIR)
    train_samples = get_train_samples(it)
    for f in files:
        compressed_chrom_mats = np.load(os.path.join(WINDOWS_COUNT_DIR, f))
        n_chunks = len(compressed_chrom_mats.files)
        for i in range(len(sigs_indices_iteration)):
            indices = sigs_indices_iteration[i]
            sig = sigs[indices]
            chrom = f[5:-4]
            mats = []
            for j in range(n_chunks):
                chrom_mat = compressed_chrom_mats[f'my_array{j}']
                chrom_mat = chrom_mat[train_samples, :, :]
                if labels:
                    chrom_mat = calculate_clustered_matrix(chrom_mat, labels)
                projection_scores = scoring_func(chrom_mat, sig, sig_names[i], it)
                mats.append(projection_scores)
            mat = np.concatenate(mats, axis=len(mats[0].shape) - 1)
            np.savez_compressed(
                os.path.join(out_dir, "score_chrom" + chrom + "_" + str(sig_names[i]) + "_it" + str(it) + ".npz"),
                my_array=mat)
            print("successfully saved score_chrom" + chrom + "_" + str(sig_names[i]) + ".npz")


def run_score_by_chunks_iteration_better(data_manager, sigs, scoring_func, additional_args, tested_sig):
    prev_chrom = -1
    # working on each set of windows separately
    dfs = []
    for key in data_manager.window_index:
        (chrom, batch_index) = eval(key)
        if chrom != prev_chrom:
            print("currently processing chrom " + chrom)
            prev_chrom = chrom
            chunk = data_manager.get_batch(chrom, batch_index)[data_manager.samples].astype('int')
            scores = scoring_func(chunk, sigs, *additional_args)
            df = pd.DataFrame({"score":scores, "indices":np.arange(chunk.shape[1])})
            df["chrom"] = chrom
            df["batch"] = batch_index
            dfs.append(df)
    df_tot = pd.concat(dfs)
    df_tot.to_csv(os.path.join(PROJECTION_SCORES_DIR,"projection_score_binary_sig"+str(tested_sig)+".csv"), index=False)


def run_score_by_chunks_iteration(files, it, sigs, scoring_func, additional_args, out_dir, file_extension, labels=None):
    train_samples = get_train_samples(it)
    for f in files:
        compressed_chrom_mats = np.load(os.path.join(WINDOWS_COUNT_DIR, f))
        n_chunks = len(compressed_chrom_mats.files)
        chrom = f[5:-4]
        mats = []
        for j in range(n_chunks):
            chrom_mat = compressed_chrom_mats[f'my_array{j}']
            chrom_mat = chrom_mat[train_samples, :, :]
            if labels is not None:
                chrom_mat = calculate_clustered_matrix(chrom_mat, labels)
            scores = scoring_func(chrom_mat, sigs, *additional_args)
            mats.append(scores)
        mat = np.concatenate(mats, axis=len(mats[0].shape) - 1)
        np.savez_compressed(
            os.path.join(out_dir, "score_chrom" + chrom + "_" + str(file_extension) + "_it" + str(it) + ".npz"),
            my_array=mat)
        print("successfully saved score_chrom" + chrom + "_" + str(file_extension) + ".npz")


def get_reset_top_values(n):
    best_score = []
    best_arg_count_mat = []
    best_chrom = []
    best_argmin = []
    for i in range(n):
        best_score.append(np.inf)
        best_chrom.append(None)
        best_argmin.append(None)
    best_arg_count_mat = np.full((25, 512, 96), 0)
    return (np.array(best_score), np.array(best_arg_count_mat),
            np.array(best_chrom), np.array(best_argmin))


def find_smallest_n_after_sorting_intersection(cur_topn, j_indices, count_matrices_cur, chrom_cur, batch_topn,
                                               i_indices,
                                               count_matrices_batch, chrom_batch, n):
    # Find the intersection indices and values
    concatenated_indices = np.concatenate((j_indices, i_indices))
    concatenated_scores = np.concatenate((cur_topn, batch_topn))
    concatenated_count_mat = np.concatenate((count_matrices_cur, count_matrices_batch))
    concatenated_chrom = np.concatenate((chrom_cur, chrom_batch))
    # Sort the intersection values and corresponding indices
    sorted_indices = np.argsort(concatenated_scores)
    sorted_concatenated_scores = concatenated_scores[sorted_indices]
    sorted_concatenated_indices = concatenated_indices[sorted_indices]
    sorted_concatenated_count_mat = concatenated_count_mat[sorted_indices]
    sorted_concatenated_chrom = concatenated_chrom[sorted_indices]

    return (sorted_concatenated_scores[:n], sorted_concatenated_indices[:n],
            sorted_concatenated_count_mat[:n], sorted_concatenated_chrom[:n])


def run_score_by_chunks_windows_greedy_aggregation(files, it, sigs, scoring_func, additional_args,
                                                   n_windows, ascending, output_name, n_windows_per_iteration):
    train_samples = get_train_samples(it)
    aggregated_windows = np.zeros(shape=(len(train_samples), 96))
    res = []
    for i in range(n_windows // n_windows_per_iteration):
        best_score, best_arg_count_mat, best_chrom, best_argmin = get_reset_top_values(n_windows_per_iteration)
        for f in files:
            chrom = f[5:-4]
            relevant_indices = [tup[1] for tup in res if tup[0] == chrom]
            compressed_chrom_mats = np.load(os.path.join(WINDOWS_COUNT_DIR, f))
            n_chunks = len(compressed_chrom_mats.files)
            index = 0
            for j in range(n_chunks):
                chrom_mat = compressed_chrom_mats[f'my_array{j}']
                chrom_mat = chrom_mat[train_samples, :, :]
                relevant_indices_in_chunk = [i - index for i in relevant_indices if
                                             index <= i < index + chrom_mat.shape[1]]

                scores = scoring_func(chrom_mat, aggregated_windows, sigs, *additional_args)
                if not ascending:
                    scores = -scores
                scores[relevant_indices_in_chunk] = np.inf
                top_args_batch = np.argsort(scores)[:n_windows_per_iteration]
                top_scores_batch = scores[top_args_batch]
                top_count_mat_batch = np.stack([chrom_mat[:, j, :] for j in top_args_batch], axis=0)
                chrom_batch = np.array([chrom for _ in range(n_windows_per_iteration)])
                best_score, best_argmin, best_arg_count_mat, best_chrom = (
                    find_smallest_n_after_sorting_intersection(best_score, best_argmin, best_arg_count_mat, best_chrom,
                                                               top_scores_batch, top_args_batch + index,
                                                               top_count_mat_batch, chrom_batch,
                                                               n_windows_per_iteration))
                index += chrom_mat.shape[1]
            print("iterated chromosome " + str(chrom))
        aggregated_windows += best_arg_count_mat.sum(axis=0)
        print(aggregated_windows.sum())
        if not ascending:
            best_score = -best_score
        res.extend([(best_chrom[i], best_argmin[i], best_score[i]) for i in range(n_windows_per_iteration)])
    res_df = pd.DataFrame(res, columns=["chrom", "window_index", "score"])
    res_df.to_csv(os.path.join(SELECTED_WINDOWS, output_name))


def run_projection_score_exp(sigs, sig_names, it, data_manager):
    for i in range(sigs.shape[0]):
        sig = sigs[i]
        run_score_by_chunks_iteration_better(data_manager,sig, batch_score_calculation_euclidean_norm, [i, it],sig_names[i])
        #run_score_by_chunks_iteration(files, it, sig, batch_score_calculation_euclidean_norm, [sig_names[i], it],
        #                              PROJECTION_SCORES_DIR, "sig_" + str(sig_names[i]))

    # calculate_projection_scores(sigs, sig_names, batch_score_calculation_euclidean_norm, list(range(sigs.shape[0])),
    #                            PROJECTION_SCORES_DIR, it)


def run_nnls_score_exp(sigs, extension, it):
    files = os.listdir(WINDOWS_COUNT_DIR)
    run_score_by_chunks_iteration(files, it, sigs, calculate_nnls_per_window, [], NNLS_SCORES_DIR, extension)
    # calculate_projection_scores(sigs, sig_names, calculate_nnls_per_window, [list(range(sigs.shape[0]))],
    #                             NNLS_SCORES_DIR, it)


def run_mle_exp(sigs, sig_names, it):
    calculate_projection_scores(sigs, sig_names, mle_with_em_per_window, [list(range(sigs.shape[0]))], MMM_SCORES_DIR,
                                it)


def run_nmf_exp(sigs, it, scoring_func, extension, labels=None):
    files = os.listdir(WINDOWS_COUNT_DIR)
    run_score_by_chunks_iteration(files, it, sigs, calculate_nmf_per_window, [scoring_func], NMF_SCORES_DIR, extension,
                                  labels)
    # calculate_projection_scores(sigs, sig_names, calculate_nmf_per_window, [list(range(sigs.shape[0]))], NMF_SCORES_DIR,
    #                             it)


def run_aggregated_nmf_exp(sigs, it, scoring_func, extension):
    files = os.listdir(WINDOWS_COUNT_DIR)
    if not scoring_func:
        ascending = False
    else:
        ascending = True
    run_score_by_chunks_windows_greedy_aggregation(files, it, sigs, calculate_nmf_per_window, [scoring_func], 250,
                                                   ascending,
                                                   "nmf_greedy_aggregation_" + extension + ".csv", 25)


if __name__ == "__main__":
    # recreation test
    data_manager = create_window_manager_object(os.path.join(EXP_DIR, "windows_index_dict.json"), 1, train=True)
    sigs = get_mutational_signatures([0,1,2,4,5])
    run_projection_score_exp(sigs, [1, 2, 3, 5, 6], 1, data_manager)

    # sigs = get_mutational_signatures([0, 4])
    # print("nmf")
    # labels_files = [os.path.join(EXP_DIR, "5_clusters.npy"), os.path.join(EXP_DIR, "10_clusters.npy"),
    #                 os.path.join(EXP_DIR, "25_clusters.npy"), os.path.join(EXP_DIR, "40_clusters.npy"),
    #                 os.path.join(EXP_DIR, "50_clusters.npy")]
    # for label_file in labels_files:
    #     labels = np.load(label_file)
    #     run_nmf_exp(sigs, 1, 0, "sig1sig5_"+label_file[90:-4], labels)
    # run_aggregated_nmf_exp(sigs, 1, 0, "sig1sig5")
    # run_nmf_exp(sigs, ["sig1_sig5"], 1)
    # print("nnls")
    # run_nnls_score_exp(sigs, ["sig1_sig5"], 1)
    # run_projection_score_exp(sigs, ["sig1","sig5"], 1)
    # calculate_projection_scores(sigs, [1, 5], )

    # should assert that attr for projection score equals that of the arrays
