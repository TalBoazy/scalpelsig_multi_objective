import numpy as np
from src.genetic_algorithm_basic_scheme import WindowsDataManager
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix, vstack, hstack
from scipy import sparse
import json
import os
from src.constants import *
from src.run_exps import return_compressed_chromosomes
from src.utils import get_train_samples
from collections import defaultdict


# mini batch k means

class ClusterMiniBatches(MiniBatchKMeans):

    def __init__(self,
                 n_clusters=8,
                 *,
                 init="k-means++",
                 max_iter=100,
                 batch_size=1024,
                 verbose=0,
                 compute_labels=True,
                 random_state=None,
                 tol=0.0,
                 max_no_improvement=10,
                 init_size=None,
                 n_init="auto",
                 reassignment_ratio=0.01,
                 ):
        super().__init__(n_clusters=n_clusters,
                         init=init,
                         max_iter=max_iter,
                         batch_size=batch_size,
                         verbose=verbose,
                         compute_labels=compute_labels,
                         random_state=random_state,
                         tol=tol,
                         max_no_improvement=max_no_improvement,
                         init_size=init_size,
                         n_init=n_init,
                         reassignment_ratio=reassignment_ratio,
                         )

        pass


#############################################################






data = csr_matrix([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
zero_rows = np.where((data != 0).sum(axis=1) == 0)[0]
b = data.data
c = np.log(b)
d = np.log(data)


sparse_data = numpy_data_transform()

nonzero_counts = sparse_data.getnnz(axis=1)

# Count rows with more than 100 non-zero elements
count = np.sum(nonzero_counts > 100)
binary_mask = sparse_data != 0

# Initialize a defaultdict to count occurrences
row_counts = {}

# Iterate over the rows of the binary mask
for row_idx in range(binary_mask.shape[0]):
    row = binary_mask.getrow(row_idx).toarray()[0]  # Get the row as a dense array
    row_tuple = tuple(row)  # Convert the row to a tuple for hashing
    if row_tuple not in row_counts:
        row_counts[row_tuple]=[]
    row_counts[row_tuple].append(row_idx) # Increment the count for this row

cluster_variances = {}
for cluster_label, indices in row_counts.items():
    # Extract rows assigned to the cluster
    cluster_rows = sparse_data[indices]

    # Compute variance along the columns (features)
    col_variances = np.asarray(cluster_rows.power(2).mean(axis=0) - np.square(cluster_rows.mean(axis=0)))

    # Compute the overall variance for the cluster
    cluster_variance = col_variances.mean()

    # Store the variance for the cluster
    cluster_variances[cluster_label] = cluster_variance
print("doneee")
kmeans = MiniBatchKMeans(n_clusters=250, init="k-means++",
                 max_iter=1000,
                 batch_size=5000,
                 verbose=1)
# Fit MiniBatchKMeans to the sparse matrix
kmeans.fit(sparse_data[:,:96])

# Get cluster assignments
cluster_labels = kmeans.labels_

from collections import Counter

# Assuming you have `cluster_labels` after fitting MiniBatchKMeans

# Count the number of data points in each cluster
cluster_sizes = Counter(cluster_labels)

# Print the size of each cluster
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: {size} data points")

inertia = kmeans.inertia_
print("Inertia (within-cluster variance):", inertia)