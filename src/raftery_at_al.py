import numpy as np
import pandas as pd
from src.MIX.Mix import Mix
from src.genetic_algorithm_basic_scheme import WindowsDataManager


def calculate_p_chosen_data(data, M):
    return M.log_likelihood(data)


def calculate_p_unchosen_data_given_data(data, M1, M2):
    exposures = M1.weighted_exposures(data)
    prev_pi = M2.pi
    M2.pi = exposures
    ll = M2.log_likelihood(data)
    M2.pi = prev_pi
    return ll


def run_exp(chosen_total, model_known_exposures, model_unknown_exposures, window_manager,
            train_samples):
    chosen_panel = np.zeros(shape=(len(train_samples), 96))
    selected = []
    for i in range(chosen_total//10):
        #debug mode
        # if i >2:
        #     break
        print("iteration number{}/{}".format(i,chosen_total//10))
        scores = []
        last_chrom=None
        for key in window_manager.window_index:
            (chrom, batch_index) = eval(key)
            # debug mode
            # if key == "('1', 2)":
            #     break
            if chrom != last_chrom:
                print("chromosome {}".format(chrom))
                last_chrom = chrom
            batch = window_manager.get_batch(chrom, batch_index)[train_samples]
            for index in range(batch.shape[1]):
                if (chrom, batch_index, index) in selected:
                    continue
                window = batch[:, index, :]
                if not window.sum():
                    continue
                if not i:
                    good_indices = np.nonzero(np.any(window, axis=1))[0].shape[0]
                    if not good_indices < 20:
                        continue
                    scores_window = calculate_p_chosen_data(window, model_known_exposures)
                    scores_window /= good_indices
                else:
                    p_combined_data = calculate_p_chosen_data(window + chosen_panel,
                                                                           model_known_exposures)
                    p_unselected_data = calculate_p_unchosen_data_given_data(window,
                                                                             model_unknown_exposures, model_known_exposures)
                    scores_window = p_combined_data / (cur_scores_window * p_unselected_data)

                    if scores_window is None or scores_window != scores_window:
                        print("yippy1")
                    if scores_window == -np.inf:
                        print("yippy2")
                    if scores_window == np.inf:
                        print("yippy3")
                scores.append([chrom, batch_index, index, scores_window])
        scores = np.array(scores)

        best_indices = np.argsort(scores[:,3].astype(np.float32))[::-1]
        values_to_add = []
        for i in best_indices[:10]:
            selected.append(tuple(scores[i][:3]))
            values_to_add.append(tuple(scores[i][:3]))
        chosen_panel += window_manager.get_entry_with_all_details(values_to_add)[train_samples]
        cur_scores_window = calculate_p_chosen_data(chosen_panel, model_known_exposures)
        model_unknown_exposures.set_data(chosen_panel)
        model_unknown_exposures.pi = None
        model_unknown_exposures._fit(["pi", "w"])
    return selected



