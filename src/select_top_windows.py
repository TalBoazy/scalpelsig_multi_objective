import numpy as np
import os
import pandas as pd
from src.utils import get_train_samples, get_positive_and_negative_samples, get_window_score_files, get_signatures_frequencies
from src.constants import *


def get_chrom_index(file_name):
    first_sep = file_name.find("_")
    second_sep = file_name.find("_", first_sep + 1)
    return file_name[11:second_sep]


# the function will iterate all the score matrices, the input will be a name t ask from utils the list of
# files to iterate over, and utils will bring the selected train test of the experiment. the method will calculate the scores for
# all windows, sort them, and save in a folder
def select_windows_projection(output_folder, tested_score, filename, it):
    windows_projection_scores_files = get_window_score_files(tested_score, PROJECTION_SCORES_DIR)
    train_samples = get_train_samples(it)
    positive, negative = get_positive_and_negative_samples(tested_score, it)
    final_scores_df = []
    for f in windows_projection_scores_files:
        chrom = get_chrom_index(f)
        scores = np.load(os.path.join(PROJECTION_SCORES_DIR, f))["my_array"][train_samples]
        final_scores = scores[positive].sum(axis=0) - scores[negative].sum(axis=0)
        chrom_final_score_df = pd.DataFrame(
            {"chrom": [chrom] * final_scores.shape[0], "window_index": list(range(final_scores.shape[0])),
             "score": final_scores})
        final_scores_df.append(chrom_final_score_df)
    total_scoring_df = pd.concat(final_scores_df, it)
    total_scoring_df.to_csv(os.path.join(output_folder, filename + ".csv"))


def select_windows(tested_score, scores_dir, panel_size, ascending):
    windows_projection_scores_files = get_window_score_files(tested_score, scores_dir)
    scores_df = []
    for f in windows_projection_scores_files:
        chrom = get_chrom_index(f)
        scores = np.load(os.path.join(scores_dir, f))["my_array"]
        chrom_score_df = pd.DataFrame(
            {"chrom": [chrom] * scores.shape[0], "window_index": list(range(scores.shape[0])),
             "score": scores})
        scores_df.append(chrom_score_df)
    total_scoring_df = pd.concat(scores_df)
    total_scoring_df = total_scoring_df.sort_values(by='score', ascending=ascending)
    top_windows_scoring = total_scoring_df.head(panel_size)
    return top_windows_scoring


def select_windows_df(result_df, to_select):
    selected = result_df.sort_values(by='score', ascending=False).head(to_select)
    return selected


def select_windows_sparsity_penalty(result_df, alpha=0.0001):
    result_df["score"] = np.exp(result_df["probabilities"]) + alpha*result_df["counts"]
    cluster_dfs = result_df.groupby("clusters")
    res = []
    portion = 250//len(cluster_dfs)
    for cluster, df in cluster_dfs:
        selected = df.sort_values(by='score', ascending=False).head(portion)
        res.append(selected)
    return pd.concat(res)


dfs = ["projection_score_binary_sig1.csv","projection_score_binary_sig2.csv", "projection_score_binary_sig3.csv",
       "projection_score_binary_sig5.csv", "projection_score_binary_sig6.csv"]

# select by proportion exp

freqs = get_signatures_frequencies([1,2,3,4,5],1)
n = 250
proportions = []
for i in range(len(freqs)):
    proportions.append(round(n*freqs[i]))
proportions[0] -= 1
a =5
dfs_selected = []
i = 0
for p in dfs:
    df = pd.read_csv(os.path.join(PROJECTION_SCORES_DIR,p))
    selected = select_windows_df(df,proportions[i])
    dfs_selected.append(selected)
    i +=1
dfs_tot = pd.concat(dfs_selected)
dfs_tot.to_csv(os.path.join(SELECTED_WINDOWS,"NNLS_binary_model_sigs_1_2_3_5_6_it1_proportion.csv"))

res_df = pd.read_csv(os.path.join(RESULTS, "mix_for_windows_exp.csv"))
res_df = res_df[res_df["counts"]>0]
alphas= [0.0001, 1e-5, 1e-10, 1e-20, 1e-30]
for a in alphas:
    selected = select_windows_sparsity_penalty(res_df, alpha=a)
    selected.to_csv(os.path.join(SELECTED_WINDOWS, "mix_for_windows_exp_"+'{:.0e}'.format(a)+".csv"), index=False)


filnames = ["NNLS_binary_model_sig_1_it1_250.csv", "NNLS_binary_model_sig_2_it1_250.csv",
            "NNLS_binary_model_sig_6_it1_250.csv","NNLS_binary_model_sig_5_it1_250.csv",
            "NNLS_binary_model_sig_3_it1_250.csv"]

tested_scores = ["sig_1_it1", "sig_2_it1", "sig_6_it1", "sig_5_it1", "sig_3_it1"]

scoring_dfs=[]
for i in range(5):
    top_scoring = select_windows(tested_scores[i], PROJECTION_SCORES_DIR, 50, False)
    scoring_dfs.append(top_scoring)
final_scoring_df = pd.concat(scoring_dfs)
final_scoring_df.to_csv(os.path.join(SELECTED_WINDOWS, "NNLS_binary_model_sigs_1_2_3_5_6_it1.csv"))

#
# filename = "NNLS_binary_model_sig1sig5_it1_500.csv"
# top_windows_scoring_sig1 = select_windows( "sig1",
#                 PROJECTION_SCORES_DIR, 250, False)
# top_windows_scoring_sig1.to_csv(os.path.join(SELECTED_WINDOWS, "NNLS_binary_model_sig1_it1.csv"))
# top_windows_scoring_sig5 = select_windows("sig5",
#                PROJECTION_SCORES_DIR, 250, False)
# top_windows_scoring_sig1.reset_index(inplace=True)
# top_windows_scoring_sig5.reset_index(inplace=True)
# top_windows_scoring = pd.concat([top_windows_scoring_sig1, top_windows_scoring_sig5])
# top_windows_scoring.to_csv(os.path.join(SELECTED_WINDOWS, filename))
# filename = "NNLS_multi_dim_model_sig1sig5_it1_normalized.csv"
# top_windows_scoring=select_windows("sig1_sig5",
#                 NNLS_SCORES_DIR, 250, True)
#top_windows_scoring.to_csv(os.path.join(SELECTED_WINDOWS, filename))
# filename = "NMF_sig1sig5_it1_kl_measured.csv"
# top_windows_scoring=select_windows("sig1_sig5",
#                NMF_SCORES_DIR, 250, True)
#top_windows_scoring.to_csv(os.path.join(SELECTED_WINDOWS, filename))
# labels = ["_5_clusters", "_10_clusters", "_25_clusters", "_40_clusters",
#                     "_50_clusters"]
# for label in labels:
#     filename = "NMF_sig1sig5_it1_"+label+".csv"
#     extension = label
#     top_windows_scoring = select_windows(extension, NMF_SCORES_DIR,250, False)
#     top_windows_scoring.to_csv(os.path.join(SELECTED_WINDOWS, filename))

# filename = "NNLS_binary_model_sig2_it1.csv"
# top_windows_scoring_sig2 = select_windows( "sig_2",
#                 PROJECTION_SCORES_DIR, 250, False)
# top_windows_scoring_sig2.to_csv(os.path.join(SELECTED_WINDOWS, filename))