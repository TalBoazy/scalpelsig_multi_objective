import numpy as np
import pandas as pd
from data.preprocess_data import create_count_matrix


def create_range_list(coordinates):
    return list(zip(coordinates['start'], coordinates['end']))


def sample_panel_data(data, panel_coordinates):
    def check_range(loc, range_list):
        for range_ in range_list:
            if range_[0] <= loc <= range_[1]:
                return True
        return False

    panel_filtered_dfs = []
    data_by_chrom = data.groupby("chrom")
    coordinates_by_chrom = panel_coordinates.groupby("chrom")
    for chrom, df in data_by_chrom:
        df_coordinates = coordinates_by_chrom.get_group(chrom)
        coordinates_dict = create_range_list(df_coordinates)

        filtered_df = df[df["loc"].apply(lambda x: check_range(x, coordinates_dict))]
        panel_filtered_dfs.append(filtered_df)
    filtered_df = pd.concat(panel_filtered_dfs)
    return filtered_df


def sample_panel_and_create_count_matrix(data, panel_coordinates):
    filtered_df = sample_panel_data(data, panel_coordinates)
    count_mat = create_count_matrix(filtered_df)
    return count_mat
