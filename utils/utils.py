# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import config as cfg


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def calc_dist(x1, y1, x2, y2):
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist


def generate_xy_org():
    df_samples = pd.read_csv(cfg.f_df_samples)
    y = np.array(df_samples[cfg.target_var_name])
    x_cnn_common = load_pickle(filename=cfg.f_data_DL_common)
    x_cnn_t_c = load_pickle(filename=cfg.f_data_DL_terrain_climate)
    x_ts_lsp = load_pickle(filename=cfg.f_data_DL_lsp)
    # x_cnn_bands_t_c = np.concatenate([x_cnn_common, x_cnn_t_c], axis=2)
    x_cnn_bands_t_c = []
    return x_cnn_common, x_cnn_t_c, x_ts_lsp, x_cnn_bands_t_c, y

def generate_xy():
    df_samples = pd.read_csv(cfg.f_df_samples)
    y = np.array(df_samples[cfg.target_var_name])
    x_spa = load_pickle(filename=cfg.f_data_spa)
    x_tempo = load_pickle(filename=cfg.f_data_tempo)
    return x_spa, x_tempo, y
