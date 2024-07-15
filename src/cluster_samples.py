from sklearn.cluster import KMeans
import numpy as np
from src.utils import get_train_samples, get_real_exposures
from data.data_utils import get_mutational_signatures
from src.constants import *
import os


if __name__ == "__main__":
    #sigs = get_mutational_signatures(np.array([1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]) - 1)
    train_samples = get_train_samples(1)
    data = get_real_exposures(train_samples, np.array(range(12)))
    n_clusters = [5, 10, 25, 40, 50]
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(data)
        cluster_labels = kmeans.labels_
        # variance calc
        np.save(os.path.join(EXP_DIR, "{}_clusters.npy".format(n)), cluster_labels)
