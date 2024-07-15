import numpy as np
import random
import pandas as pd
from collections import defaultdict
import itertools


class WindowsDataManager:

    def __init__(self, n_debug = None):
        self.chromosomes = None
        self.window_index = None
        self.batch_size = None
        self.n_samples = None
        self.samples = None
        self.n_features = n_debug

    def update_vals(self, compressed_chroms, window_index, batch_size, n_samples, samples):
        self.chromosomes = compressed_chroms
        self.window_index = window_index
        if self.n_features is not None:
            self.window_index = {k: self.window_index[k] for k in itertools.islice(self.window_index, self.n_features)}
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.samples = samples


    def get_batch(self, chrom, i):
        return self.chromosomes[chrom]["my_array{}".format(i)]


    def get_entry_with_all_details(self, detailed_indices):
        features_chromosome = np.zeros(shape=(self.n_samples, 96))
        for i in detailed_indices:
            (chrom, batch_index, index) = i
            batch = self.get_batch(chrom, batch_index)
            features_chromosome += batch[:,int(index),:]
        return features_chromosome


    def get_entries_(self, indices, chrom, batch):
        df = pd.DataFrame({"i":indices, "c":chrom, "b":batch})
        batch_groups = df.groupby(["c","b"])
        res = np.zeros(shape=(self.n_samples, 96))
        for b, i in batch_groups:
            batch = self.get_batch(b[0],b[1])
            res += batch[self.samples][:,i["i"]].sum(axis=1)
        return res

    def get_entry(self, indices):
        features_chromosome = np.zeros(shape=(self.n_samples, indices.shape[0], 96))
        i_chrom = np.zeros(shape=(indices.shape[0])).astype(np.int64)
        for key in self.window_index:
            start = self.window_index[key][0]
            end = self.window_index[key][1]
            (chrom, batch) = eval(key)
            # it is not a waste because all those windows are mapped and will be called eventually
            batch_mat = self.chromosomes[chrom]["my_array{}".format(batch)]
            # iterating each of the chromosomes
            for i in range(indices.shape[0]):
                # finished constructing the chromosome
                if i_chrom[i] == indices.shape[1]:
                    continue
                if start <= indices[i, i_chrom[i]] < end:
                    # find j
                    j = i_chrom[i] + 1
                    while j < indices.shape[1] and start <= indices[i, j] < end:
                        j += 1
                    index_in_batch = indices[i, i_chrom[i]:j] - start
                    features_chromosome[:, i, :] += batch_mat[:, index_in_batch,
                                                    :].sum(axis=1)
                    i_chrom[i] = j
        print("successfully loaded all the chromosomes")
        return features_chromosome

    def get_entries(self, indices):
        return self.get_entry(indices)


    def construct_panel_from_indices(self, indices):
        panel = np.zeros(shape=(len(indices),self.n_samples, 96))
        # create a grouped dict by keys
        key_dicts = []
        for c in range(len(indices)):
            key_dict = defaultdict(list)
            for tup in indices[c]:
                key_dict[tup[0]].append(tup[1])
            key_dicts.append(key_dict)

        for k in self.window_index:
            batch = None
            for i in range(len(key_dicts)):
                if k in key_dicts[i]:
                    if batch is None:
                        print(k)
                        chrom, i = eval(k)
                        batch = self.get_batch(chrom, i)[self.samples]
                    chosen = batch[:, key_dicts[i][k], :]
                    panel[i] += chosen.sum(axis=1)
        return panel




    def transform_index_to_index_by_chrom(self, score, indices):
        tuple_list = []
        for i in range(len(indices)):
            for key in self.window_index:
                start = self.window_index[key][0]
                end = self.window_index[key][1]
                (chrom, batch) = eval(key)
                if start <= indices[i] < end:
                    fitted_index = indices[i] - start
                    tuple_list.append((chrom, fitted_index, score))
        score_df = pd.DataFrame(tuple_list, columns=["chrom", "index", "score"])
        return score_df


class BasicGeneticAlgorithm:

    def __init__(self, n_generations, n_children, inner_mut_prob, outer_mut_prob, scoring_func, n_selected,
                 n_samples, n_parents):
        self.n_features = None
        self.n_generations = n_generations
        self.n_children = n_children
        self.inner_p = inner_mut_prob
        self.outer_p = outer_mut_prob
        self.fitness = scoring_func
        self.initial_partition = None
        self.chromosome_size = n_selected
        self.n_samples = n_samples
        self.n_parents = n_parents
        self.window_manager = WindowsDataManager()
        pass

    def split_to_populations(self, population_indices, initial_partition=None):
        self.n_features = len(population_indices)
        if initial_partition is not None:
            self.initial_partition = initial_partition
        else:
            population_members = population_indices
            random.shuffle(population_members)
            self.initial_partition = np.split(np.array(population_members),
                                              len(population_members) // self.chromosome_size)

    def calculate_fittness(self, chromosome):
        return self.fitness(chromosome)

    def store_data(self, compressed_chroms, window_index, batch_size, n_samples):
        self.window_manager.update_vals(compressed_chroms, window_index, batch_size, n_samples)

    def calculate_fittness_population(self, population):
        chromosomes = self.window_manager.get_entries(np.sort(np.array(population), axis=1))
        # for i in range(len(population)):
        #     chromosome = self.window_manager.get_entries(np.sort(np.array(population[i])))
        scores = self.calculate_fittness(chromosomes)
        return scores

    def choose_the_best_parents(self, scores, n, greedy=True):
        if greedy:
            print(scores.min())
            return scores.argsort()[:n]
        else:
            # todo if we continue with this direction
            pass

    def recombination(self, chrom_a, chrom_b):
        recombinant = np.random.randint(2, size=self.chromosome_size)
        new_chrom = np.where(recombinant, chrom_a, chrom_b)
        return new_chrom

    def assign_parents_to_pairs(self, parents):
        random.shuffle(parents)
        pairs = []
        num_parents = len(parents)
        for i in range(0, num_parents - 1, 2):
            pairs.append((parents[i], parents[i + 1]))
        return pairs

    def evolution(self, scores, population):
        parents_to_select = int(scores.shape[0] * self.n_parents)
        if parents_to_select % 2:
            parents_to_select += 1
        parents = self.choose_the_best_parents(scores, parents_to_select)
        parents_pair = self.assign_parents_to_pairs(parents)
        new_generation = []
        for pair in parents_pair:
            chrom_a, chrom_b = population[pair[0]], population[pair[1]]
            for i in range(self.n_children):
                new_chrom = self.recombination(chrom_a, chrom_b)
                new_generation.append(new_chrom)
        return new_generation

    def fit_within_sub_population(self, initial_population):
        population = initial_population
        for i in range(self.n_generations):
            print("generation {}".format(i))
            scores = self.calculate_fittness_population(population)
            best_score_chrom = scores.argmin()
            best_score = scores.min()
            best_chrom = population[best_score_chrom]
            chrom_df = self.window_manager.transform_index_to_index_by_chrom(best_score,best_chrom)
            chrom_df.to_csv("generation_{}_score_{:.3f}.csv".format(i,best_score))
            population = self.evolution(scores, population)
        best_score_chrom = scores.argmin()
        best_chrom = population[best_score_chrom]
        return best_chrom, best_score_chrom

    def fit(self, compressed_chroms, window_index, batch_size, n_samples, population_indices,
            initial_partition=None):
        self.store_data(compressed_chroms, window_index, batch_size, n_samples)
        self.split_to_populations(population_indices, self.initial_partition)
        # new_population = []
        # while len(population_indices) > self.population_size:
        #     for i in self.initial_partition:
        #         best_chrom = self.fit_within_sub_population(initial_partition[i])
        #         new_population.extend(best_chrom)
        best_chrom, best_score = self.fit_within_sub_population(self.initial_partition)
        res = self.window_manager.transform_index_to_index_by_chrom(best_score, best_chrom)
        return res
