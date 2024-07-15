import numpy as np
import pandas as pd
from data.data_utils import write_list_to_file


class ParetoFront:

    def sort_objective_dictionary(self, obj):
        sorted_obj = sorted(obj.items(), key=lambda x: x[1], reverse=True)
        sorted_keys = [item[0] for item in sorted_obj]
        sorted_scores = [item[1] for item in sorted_obj]
        return sorted_keys, sorted_scores

    def check_non_dominanted(self, obj1, obj2, solutions, win):
        elements_to_remove = []
        for s in solutions:
            if obj1[s] > obj1[win] and obj2[s] > obj2[win]:
                return False, elements_to_remove
            if obj1[s] < obj1[win] and obj2[s] < obj2[win]:
                elements_to_remove.append(s)
        return True, elements_to_remove

    def naive_pareto_front(self, obj1, obj2, n):
        """
        the naivest algorithm, fits for two objectives.
        :param obj1: dictionary for objective 1 function
        :param obj2: dictionary for objective 2 function
        :param n: the requested selections
        :return: the selected n points on the pareto-front

        * based on Mishra, K. K., & Harit, S. (2010). A fast algorithm for finding the non
          dominated set in multi objective optimization
        """
        sorted_win_obj1, sorted_score_obj1 = self.sort_objective_dictionary(obj1)
        j = 0
        pareto_optimal = []
        while j < n:
            pareto_optimal_temporal = set({})
            pareto_optimal_temporal.add(sorted_win_obj1[j])
            for i in range(j + 1, len(sorted_win_obj1)):
                p = sorted_win_obj1[i]
                if p not in pareto_optimal:
                    res, elems_to_remove = self.check_non_dominanted(obj1, obj2, pareto_optimal_temporal, p)
                    pareto_optimal_temporal.difference_update(elems_to_remove)
                    if res:
                        pareto_optimal_temporal.add(p)
                        if i == j + 1:
                            j = j + 1
                    if not len(pareto_optimal_temporal):
                        pareto_optimal_temporal.add(p)
            pareto_optimal.extend(list(pareto_optimal_temporal))
            if len(pareto_optimal) >= n:
                break
            j += 1
        return pareto_optimal[:n]

    def select_top_n(self, obj1, obj2, n):
        panel_windows = []
        i=0
        sorted_win_obj1, sorted_score_obj1 = self.sort_objective_dictionary(obj1)
        sorted_win_obj2, sorted_score_obj2 = self.sort_objective_dictionary(obj2)
        while len(panel_windows)<n:
            elems = n - len(panel_windows)
            top1 = int(elems // 2)
            top2 = elems - top1
            panel_windows.extend(sorted_win_obj1[i:i+top1])
            panel_windows.extend(sorted_win_obj2[i:i+top2])
            panel_windows = list(set(panel_windows))
            i+=max(top1,top2)
        return panel_windows

    def select_weighted_score(self, obj1, obj2, n, w1, w2):
        combined_obj = {key: w1 * obj1[key] + w2 * obj2[key] for key in obj1.keys()}
        sorted_win, sorted_score = self.sort_objective_dictionary(combined_obj)
        return sorted_win[:n]


def convert_df_to_dict(obj):
    return {row['Name']: row['Score'] for index, row in obj.iterrows()}


def add_matching_values(df1, df2):
    values_to_add = set(df2['Name']).difference(df1['Name'])
    missing_rows = pd.DataFrame({'Name': list(values_to_add), 'Score': [0] * len(values_to_add)})
    df1 = pd.concat([df1, missing_rows], ignore_index=True)
    return df1


if __name__ == "__main__":
    obj1_df = pd.read_csv(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\icgc_exp\\panel_windows_icgc_exp_it1_sig1_obj2_nwin250_06-Mar-2024_17-49.tsv",
        sep="\t")
    obj2_df = pd.read_csv(
        "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\icgc_exp\\panel_windows_icgc_exp_it1_sig3_obj2_nwin250_06-Mar-2024_18-14.tsv",
        sep='\t')
    obj1_df = add_matching_values(obj1_df, obj2_df)
    obj2_df = add_matching_values(obj2_df, obj1_df)
    obj1 = convert_df_to_dict(obj1_df)
    obj2 = convert_df_to_dict(obj2_df)

    pareto_front = ParetoFront()
    pareto_front_windows = pareto_front.naive_pareto_front(obj1, obj2, 250)
    write_list_to_file(pareto_front_windows,
                       "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\icgc_exp\\panel_windows_pareto_front.txt")
    top_n_windows = pareto_front.select_top_n(obj1, obj2, 250)
    write_list_to_file(top_n_windows,
                       "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\icgc_exp\\panel_windows_top_n.txt")
    weighted_windows = pareto_front.select_weighted_score(obj1, obj2, 250, 0.25, 0.75)
    write_list_to_file(weighted_windows,
                       "C:\\Users\\talbo\\masters\\mutational_signatures_analysis\\scalpelsig_multi_objective\\data\\icgc_exp\\panel_windows_combined_score.txt")
