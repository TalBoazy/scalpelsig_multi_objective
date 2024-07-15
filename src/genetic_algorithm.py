import numpy as np
import random


class GeneticAlgorithm:

    def __init__(self, o, initial_size=400, pool_size=0.3, mut_prob=0.2, old_gen_size=5, chromosome_size=5000, max_iter=1000, epsilon=1e-5):
        self.initial_size = initial_size
        self.pool_size = pool_size
        self.o = o
        self.data_manager = None
        self.mut_prob = mut_prob
        self.old_gen_size = old_gen_size
        self.chromosome_size = chromosome_size
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.chromosomes = []
        self.e = None
        self.n_features = None
        self.pi = None
        pass

    def set_data(self, data_manager, real_exposures, signatures):
        self.pi = real_exposures
        self.e = signatures
        self.data_manager = data_manager
        last_key = list(self.data_manager.window_index.keys())[-1]
        self.n_features = self.data_manager.window_index[last_key][1]
        self.initial_population()

    def initial_population(self):
        n = self.initial_size
        for i in range(n):
            selected = sorted(np.random.randint(0, self.n_features, size=self.chromosome_size))
            # get the keys
            chrom = []
            j = 0
            for key in self.data_manager.window_index:
                start, end = self.data_manager.window_index[key][0], self.data_manager.window_index[key][1]
                while j<len(selected) and start <= selected[j] < end:
                    if j-start <0:
                        a = 5
                    chrom.append((key, selected[j] - start))
                    j += 1
            self.chromosomes.append(chrom)


    def score_chromosome(self, panel):
        return self.o(panel, self.pi, self.e)

    def tournament_method(self):
        top_score = np.inf,
        winner_chrom = None
        n = int(self.pool_size * len(self.chromosomes))
        panels = self.data_manager.construct_panel_from_indices(self.chromosomes)
        pool = []
        while True:
            if len(pool) == n:
                break
            candidates = random.sample(range(len(self.chromosomes)), 3)
            best_score = np.inf
            winner = None
            for candit in candidates:
                score = self.score_chromosome(panels[candit])
                if score < best_score:
                    best_score = score
                    winner = candit
                if score < top_score:
                    top_score = score
                    winner_chrom = winner
            pool.append(winner)
        return pool, top_score, winner_chrom

    def crossover(self, chrA, chrB):
        selected = random.sample(range(self.chromosome_size), k=2)
        selected.sort()
        low, high = selected[0], selected[1]
        # bp1_a, bp1_b, bp2_a, bp2_b = None, None, None, None
        #
        # for i in range(self.chromosome_size):
        #     a_index = chrA[i][1]+self.data_manager.window_index[chrA[i][0]][0]
        #     b_index = chrB[i][1] + self.data_manager.window_index[chrB[i][0]][0]
        #     if low <= a_index < high and bp1_a is None:
        #         bp1_a = i
        #     elif a_index >= high and bp2_a is None:
        #         bp2_a = i
        #     if low <= b_index < high and bp1_b is None:
        #         bp1_b = i
        #     elif b_index >= high and bp2_b is None:
        #         bp2_b = i
        #     if not (bp1_a is None or bp2_a is None or bp1_b is None or bp2_b is None):
        #         break
        chrA_new = chrA[:low] + chrB[low:high] + chrA[high:]
        chrB_new = chrB[:low] + chrA[low:high] + chrB[high:]
        return chrA_new, chrB_new

    def mutation(self, chrom):
        mutations = np.random.randint(0, self.n_features, size=int(self.chromosome_size*self.mut_prob)).sort()
        pass


    def generate_new_gen(self,pool):
        new_chromosomes = []
        j=0
        for i in range(self.initial_size//2):
            parents = random.sample(pool, k=2)
            chrA, chrB = self.chromosomes[parents[0]], self.chromosomes[parents[1]]
            childA, childB = self.crossover(chrA, chrB)
            new_chromosomes.append(childA)
            new_chromosomes.append(childB)
        return new_chromosomes

    def fit(self, data_manager, real_exposures, signatures):
        self.set_data(data_manager, real_exposures, signatures)
        old_score = np.inf
        best_chrom = None
        chromosomes = self.chromosomes
        for i in range(self.max_iter):
            # calculate scores
            self.chromosomes = chromosomes
            pool, best_score, best_chrom = self.tournament_method()
            if np.abs(best_score - old_score) <= self.epsilon:
                break
            chromosomes = self.generate_new_gen(pool)
        return self.chromosomes[best_chrom]





