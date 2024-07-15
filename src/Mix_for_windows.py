from scipy import sparse
import numpy as np
from src.MIX.Mix import Mix
from src.MIX.logsumexp import logsumexp, sparse_logsumexp
import time


class MixWindows(Mix):

    def __init__(self, num_clusters, num_topics, num_words=None, init_params=None, epsilon=1e-4, max_iter=1e5,
                 chunk_size=1000):
        """

        :param num_clusters: Positive integer, number of clusters to learn.
        :param num_topics: Positive integer, number of topics to learn.
        :param init_params: Dictionary with keys e, pi, w and values for initial parameters.
        :param epsilon: Positive small number, the stop criterion for fit.
        :param max_iter: Positive integer, maximum number of iterations for fit.
        """

        super().__init__(num_clusters,
                         num_topics,
                         num_words,
                         init_params,
                         epsilon,
                         max_iter)
        self.num_windows = None
        self.num_samples = None
        self.chunk_size = chunk_size

    def refit(self, data, handle_zero_prob_mutations='remove'):
        """
        Fitting only the clusters (pi) and the weights (w).
        :param data: windowmanager object
        :return:
        """
        if self.e is None:
            raise ValueError('e was not set, can not refit')

        self.set_data(data)
        if handle_zero_prob_mutations == 'remove':
            # to avoid any problem we remove 0 prob mutations
            real_m, good_indices = self.remove_zero_prob_mutations()

            # Run refiting
            output = self._fit(['w', 'pi'])

            # Fix signatures back
            self.insert_zero_prob_mutations(data, real_m, good_indices)

            return output
        else:
            raise ValueError('not implemented other strategies yet')

    def set_data(self, data):
        self.data = data
        for key in self.data.window_index:
            (chrom, batch_index) = eval(key)
            batch = self.data.get_batch(chrom, batch_index)[self.data.samples]
            # batch = self.data.get_batch(chrom, batch_index)[self.data.samples,:20]
            self.num_samples = batch.shape[0]
            break
        self.num_words = self.e.shape[1]

    def _fit(self, params):
        if self.e is None:
            self.e = np.random.dirichlet([0.5] * self.num_words, self.num_topics)
        if self.pi is None:
            self.pi = np.random.dirichlet([0.5] * self.num_topics, self.num_clusters * self.num_samples)
            self.pi = self.pi.reshape((self.num_clusters, self.num_samples, self.num_topics))
        if self.w is None:
            self.w = np.random.dirichlet([2] * self.num_clusters)

        self.pi = np.log(self.pi)
        self.w = np.log(self.w)
        self.e = np.log(self.e)
        expected_w, log_expected_pi, log_expected_e, prev_log_likelihood = self.expectation_step()
        log_likelihood = prev_log_likelihood
        for iteration in range(self.max_iter):
            # print(iteration, log_likelihood)
            # maximization step
            self.w, self.pi, self.e = self.maximization_step(expected_w, log_expected_pi, log_expected_e, params)

            # expectation step
            expected_w, log_expected_pi, log_expected_e, log_likelihood = self.expectation_step()
            print(log_likelihood)
            if log_likelihood - prev_log_likelihood < self.epsilon:
                break

            prev_log_likelihood = log_likelihood

        self.pi = np.exp(self.pi)
        self.e = np.exp(self.e)
        self.w = np.exp(self.w)
        return log_likelihood

    def pre_expectation_step_(self, chunk, log_chunk):
        num_samples, num_clusters, num_topics, num_words, num_windows = self.num_samples, self.num_clusters, self.num_topics, self.num_words, \
            chunk.shape[0]
        log_likelihood = np.zeros(shape=(num_clusters, num_windows))
        log_expected_e = np.zeros(shape=(num_clusters, num_windows, num_topics, num_words))
        log_expected_pi = np.zeros(shape=(num_clusters, num_samples, num_windows, num_topics))
        log_e = self.e
        for l in range(num_clusters):
            log_pi = self.pi[l]
            log_prob_topic_word = (log_e.T[:, np.newaxis] + log_pi).T
            log_prob_word = logsumexp(log_prob_topic_word, axis=0)
            log_likelihood[l] = (chunk * log_prob_word[np.newaxis, :, :]).sum(axis=(1, 2))
            log_expected_e_sample = log_prob_topic_word[:, np.newaxis, ...] + log_chunk[np.newaxis, ...] - log_prob_word
            log_expected_e_sample = np.transpose(log_expected_e_sample, (2, 1, 0, 3))
            log_expected_e[l] = logsumexp(log_expected_e_sample, axis=0)
            log_expected_pi[l] = logsumexp(log_expected_e_sample, axis=3)
        return log_expected_pi, log_expected_e, log_likelihood

    def expectation_step(self):
        num_samples, num_clusters, num_topics, num_words = self.num_samples, self.num_clusters, self.num_topics, self.num_words
        log_expected_pi = np.log(np.zeros(shape=(num_clusters, num_samples, num_topics)))
        log_expected_e = np.log(np.zeros(shape=(num_topics, num_words)))
        expected_w = np.log(np.zeros(shape=(num_clusters,)))
        log_likelihood = 0
        prev_chrom = -1
        for key in self.data.window_index:
            (chrom, batch_index) = eval(key)
            if chrom != prev_chrom:
                print("currently processing chrom " + chrom)
                prev_chrom = chrom
            batch = self.data.get_batch(chrom, batch_index)[self.data.samples]
            # batch = self.data.get_batch(chrom, batch_index)[self.data.samples, :20]
            log_batch = np.log(batch)
            expected_pi_sample_cluster, expected_e_sample_cluster, likelihood_cluster_sample = self.pre_expectation_step_(
                batch.transpose((1, 0, 2)), log_batch.transpose((1, 0, 2)))
            likelihood_cluster_sample += self.w[:, np.newaxis]
            tmp = logsumexp(likelihood_cluster_sample, 0, keepdims=True)
            log_likelihood += np.sum(tmp)
            likelihood_cluster_sample -= tmp
            expected_pi_sample_cluster += likelihood_cluster_sample[:, np.newaxis, :, np.newaxis]
            expected_e_sample_cluster += likelihood_cluster_sample[:, :, np.newaxis, np.newaxis]
            log_expected_pi = logsumexp(np.concatenate(
                (logsumexp(expected_pi_sample_cluster, 2)[..., np.newaxis], log_expected_pi[..., np.newaxis]), axis=3),
                axis=3)
            log_expected_e = logsumexp(np.concatenate(
                (logsumexp(expected_e_sample_cluster, (0, 1))[..., np.newaxis], log_expected_e[..., np.newaxis]),
                axis=2), axis=2)
            expected_w = logsumexp(np.concatenate
                                   ((logsumexp(likelihood_cluster_sample, 1)[..., np.newaxis],
                                     expected_w[..., np.newaxis]), axis=1),
                                   axis=1)
        return expected_w, log_expected_pi, log_expected_e, log_likelihood

    def maximization_step(self, log_expected_w, log_expected_pi, log_expected_e, params):
        if 'w' in params:
            w = log_expected_w - logsumexp(log_expected_w)
        else:
            w = self.w
        if 'pi' in params:
            # todo check summation to 1 here
            pi = log_expected_pi - np.tile(logsumexp(log_expected_pi, axis=2, keepdims=True), (1, 1, self.num_topics))
        else:
            pi = self.pi
        if 'e' in params:
            e = log_expected_e - logsumexp(log_expected_e, axis=1, keepdims=True)
        else:
            e = self.e
        return w, pi, e

    def predict(self, data):
        """

        :param data: ndarray (N, num_words) of non-negative integers.
        :return clusters: ndarray (N,) of integers representing the clusters.
        :return topics: ndarray (N, num_topics) of non-negative integers, counts of topics in the data.
        :return probabilities: ndarray (N,), probabilities[i] = log(Pr[sample[i], topics[i], clusters[i]])
        """
        clusters_tot, prob_tot, counts_tot = [], [], []
        indices, batchs, chroms = [], [], []
        topics_clusters_samples = np.zeros(shape=(self.num_clusters, self.num_samples, self.num_topics))
        num_samples = self.num_samples
        log_e = np.log(self.e)
        log_w = np.log(self.w)
        prev_chrom = -1
        # working on each set of windows separately
        for key in data.window_index:
            (chrom, batch_index) = eval(key)
            if chrom != prev_chrom:
                 print("currently processing chrom " + chrom)
                 prev_chrom = chrom
            cur_data = data.get_batch(chrom, batch_index)[data.samples].astype('int')
            num_windows = cur_data.shape[1]
            clusters = np.zeros(num_windows, dtype='int')
            probabilites = np.log(np.zeros(num_windows))
            topics = np.zeros((num_windows, num_samples, self.num_topics), dtype='int')
            for cluster in range(self.num_clusters):
                curr_pi = np.log(self.pi[cluster])
                pr_topic_word = (log_e.T[:, np.newaxis] + curr_pi).T
                likeliest_topic_per_word = np.argmax(pr_topic_word, axis=0)
                for window in range(num_windows):
                    curr_prob = log_w[cluster]
                    curr_topic_counts = np.zeros(shape=(self.num_samples, self.num_topics), dtype='int')
                    curr_word_counts = cur_data[:, window, :]
                    if not cluster:
                        indices.append(window)
                        chroms.append(chrom)
                        batchs.append(batch_index)
                        counts_tot.append(curr_word_counts.sum())
                    for word in range(curr_word_counts.shape[1]):
                        samples = np.argwhere(curr_word_counts[:,word] != 0)[:,0]
                        if not samples.shape[0]:
                            continue
                        curr_topic_counts[np.arange(num_samples),likeliest_topic_per_word[:,word]] += cur_data[:, window, word]
                        curr_prob += (cur_data[:,window, word] * pr_topic_word[likeliest_topic_per_word[:,word], np.arange(num_samples), word]).sum()
                    if curr_prob > probabilites[window]:
                        probabilites[window] = curr_prob
                        clusters[window] = cluster
                        topics[window] = curr_topic_counts
            clusters_tot.append(clusters)
            prob_tot.append(probabilites)
            for cluster in range(self.num_clusters):
                windows = np.argwhere(clusters == cluster)
                topics_in_cluster_batch = topics[windows].sum(axis=0)
                topics_clusters_samples += topics_in_cluster_batch
        return (np.concatenate(clusters_tot), topics_clusters_samples, np.concatenate(prob_tot), np.array(counts_tot),
                np.array(indices), np.array(chroms), np.array(batchs))
