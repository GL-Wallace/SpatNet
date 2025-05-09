# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

# coding=utf-8
import os

# model hyper-parameters
device = 'cuda'  # 'cpu' or 'cuda'
rand_seed = 51

# ['SpaNet', 'TempoNet', 'SpatNet', 'MTNet']
model_name = 'MTNET'  

# hyper-parameter of SPATNet
dropout_rate = 0.25

# hyper-parameter of SpaNet
num_channels = 17


# hyper-parameter of TempoNet
tempo_input_size = 70          # feature_size of time series
tempo_hidden_size = 64         # hidden size and layers do not need to be large
tempo_num_layers = 2

# hyper-parameter of Cross Attender
num_heads = 8
hidden_dim =128

# hyper-parameter for training
lr = 1e-3
lr_min = 0.00001 

batch_size = 64
epochs = 200  # need to consider early stopping to avoid overfitting
eval_interval = 1

data_dir = './data/'
log_dir = './log/'
f_df_samples = os.path.join(data_dir, 'sample_values.csv')   # user need to assign the filename of the sample data (including columns of the target soil property, e.g. soil organic carbon values)
target_var_name = 'SOM'     # the column name for the target property (y) that needs to be predicted
f_data_spa = os.path.join(data_dir, 'spa_windows_data.pkl')
f_data_tempo = os.path.join(data_dir, 's2_3yr_7m.pkl')               # the pickle file of the input data (X) for LSTM (i.e. phenological data with temporally dynamic information)

train_test_id = 2
f_train_index = os.path.join(data_dir, 'train_test_idx', 'train_{}.pkl'.format(train_test_id))  # the pickle file of the sample id list for the training set
f_test_index = os.path.join(data_dir, 'train_test_idx', 'test_{}.pkl'.format(train_test_id))    # the pickle file of the sample id list for the testing set

best_model_path = './log/{}_{}.pth'.format(model_name, train_test_id)  # the save path of the model parameters

