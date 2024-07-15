import pandas as pd
from scipy.optimize import nnls
import numpy as np
from data.data_utils import get_mutational_signatures


def calculate_exposures(data, sig):
    exposures = []

    for m in data:
        exposures.append(nnls(sig.T, m)[0] / m.sum())

    # normalize exposures
    for i in range(len(exposures)):
        exposures[i] = exposures[i] / exposures[i].sum()
    return np.array(exposures)


def calculate_average_exposures(exposures):
    average_exposures = exposures.mean(axis=0)
    print(average_exposures)


if __name__ == "__main__":
    data = pd.read_csv(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig\\signature-estimation-py\\nz_signature_estimation.tsv",
        sep='\t')
    mat = data.iloc[:, 1:].to_numpy()

    mat[mat < 100] = 0

    # check average sum of each signature
    print(mat.sum(axis=0) / mat.sum())

    # check the strongest pairs and count them

    top_indices = np.argsort(mat, axis=1)[:, -2:]

    # Create a binary matrix with the same shape as the original array
    binary_matrix = np.zeros_like(mat)

    # Set the top 2 max elements in each row to 1
    np.put_along_axis(binary_matrix, top_indices, 1, axis=1)
    top_indices_dict = {}
    for indices in top_indices:
        indices_tuple = tuple(indices)
        if indices_tuple not in top_indices_dict:
            top_indices_dict[indices_tuple] = 0
        top_indices_dict[indices_tuple] += 1

    print("Top indices dictionary:")
    for indices, count in top_indices_dict.items():
        print(indices, ":", count)

    data = np.load(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\gbm_exp\\tcga_gbm_all_counts.npy")
    row_sums = np.sum(data, axis=1)

    # Create a boolean mask for rows with sum greater than or equal to 200
    mask = row_sums >= 200

    # Filter out rows with sum less than 200
    filtered_data = data[mask]
    sigs = get_mutational_signatures([0, 4, 10])
    exposures = calculate_exposures(filtered_data, sigs)
    calculate_average_exposures(exposures)
    np.save(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\gbm_exp\\tcga_gbm_all_exposures_200_tresh.npy",
        exposures)

    mask = row_sums >= 150

    # Filter out rows with sum less than 200
    filtered_data = data[mask]
    sigs = get_mutational_signatures([0, 4, 10])
    exposures = calculate_exposures(filtered_data, sigs)
    calculate_average_exposures(exposures)
    np.save(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\gbm_exp\\tcga_gbm_all_exposures_150_tresh.npy",
        exposures)

    mask = row_sums >= 100

    # Filter out rows with sum less than 200
    filtered_data = data[mask]
    sigs = get_mutational_signatures([0, 4, 10])
    exposures = calculate_exposures(filtered_data, sigs)
    calculate_average_exposures(exposures)
    np.save(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\gbm_exp\\tcga_gbm_all_exposures_150_tresh.npy",
        exposures)
