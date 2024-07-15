from sklearn.decomposition import non_negative_factorization
import pandas as pd
import numpy as np
from data.data_utils import get_mutational_signatures
from scipy.optimize import nnls


def calculate_nnls(X, E):
    # Transpose E to match the dimensions
    E = E.T

    # Initialize an empty array to store the results
    pi = np.zeros((X.shape[0], E.shape[1]))
    norms = np.zeros(shape=X.shape[0])
    # Iterate over each row of X
    non_zero_indices = []
    for i in range(X.shape[0]):
        # Solve non-negative least squares regression
        if X[i,:].sum():
            non_zero_indices.append(i)
        pi[i, :], norms[i] = nnls(E, X[i, :])
    pi = (pi.T / pi.sum(axis=1)).T
    return pi, norms[non_zero_indices].mean()


if __name__ == "__main__":
    #exp
    data1 = pd.read_csv("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\Biterm\\data\\ICGC-BRCA\\counts.ICGC-BRCA-EU_BRCA_22.WGS.SBS-96.tsv", sep='\t')
    X1 = data1.iloc[:, 1:].values
    e = get_mutational_signatures([0, 1, 2, 4, 5, 12, 16, 17, 19, 25, 29])
    #e = e[:5]

    pi1 = calculate_nnls(X1, e)[0]
    statistics1 = (pi1>0.1).sum(axis=0)/560
    #np.save(
    #    "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\exposures_exp.npy",
    #    pi1)
    data = pd.read_csv(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\count_mat.tsv")
    X = data.iloc[:, 1:].values
    e = get_mutational_signatures([0, 1, 2, 4, 5, 12, 16, 17, 19, 25, 29])
    e = e[:5]

    pi = calculate_nnls(X, e)[0]
    statistics = (pi>0.05).sum(axis=0)/569

    real = np.array([0.9,0.8,0.25, 0.83,0.01,0.77,0.05,0.17,0.005,0.01,0.005])
    # print(np.linalg.norm(real-statistics1))
    # print(np.linalg.norm(real - statistics))
    a = 5
    np.save("C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\icgc_exp\\nnls_exposures.npy", pi)
