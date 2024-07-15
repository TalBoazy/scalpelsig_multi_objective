from src.genetic_algorithm_basic_scheme import WindowsDataManager
import numpy as np
from src.utils import get_real_exposures
from src.constants import *



def shuffle(windows):
    def shuffle_window(window):
        np.random.shuffle(window)
        window_t = window.T
        np.random.shuffle(window_t)
        return window_t.T

    shuffled_windows = np.zeros(shape=windows.shape)
    for i in range(windows.shape[1]):
        shuffled_windows[:,i,:] = shuffle_window(windows[:,i,:])
    return shuffled_windows



