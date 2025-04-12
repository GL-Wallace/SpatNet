# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

# coding=utf-8
import os
import pickle

# model hyper-parameters
device = 'cuda'  # 'cpu' or 'cuda'
rand_seed = 188

# ['SpaNet', 'TempoNet', 'SpatNet',]
model_name = 'TempoNet'  

# hyper-parameter of SpaNet
# num_channels = 10
num_channels = 1 

# hyper-parameter of TempoNet (small values for parameters for initializing the model training)
tempo_input_size = 70          # feature_size of time series
tempo_hidden_size = 32         # hidden size and layers do not need to be large
tempo_num_layers = 2
tempo_dropout = 0.2

# hyper-parameter for training
lr = 0.1
lr_min = 0.00001 

batch_size = 32
epochs = 100    # need to consider early stopping to avoid overfitting
eval_interval = 1

data_dir = './data/'
log_dir = './log/'
f_df_samples = os.path.join(data_dir, 'sample_values.csv')   # user need to assign the filename of the sample data (including columns of the target soil property, e.g. soil organic carbon values)
target_var_name = 'SOM'     # the column name for the target property (y) that needs to be predicted
f_data_spa = os.path.join(data_dir, 's2_3yr_7m.pkl')
f_data_tempo = os.path.join(data_dir, 's2_3yr_7m.pkl')               # the pickle file of the input data (X) for LSTM (i.e. phenological data with temporally dynamic information)

train_test_id = 1
f_train_index = os.path.join(data_dir, 'train_test_idx', 'train_{}.pkl'.format(train_test_id))  # the pickle file of the sample id list for the training set
f_test_index = os.path.join(data_dir, 'train_test_idx', 'test_{}.pkl'.format(train_test_id))    # the pickle file of the sample id list for the testing set

model_save_pth = './log/{}_{}.pth'.format(model_name, train_test_id)  # the save path of the model parameters

