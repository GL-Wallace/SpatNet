# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import pickle
import random

output_dir = '/mnt/e/PythonPractice/SpatNet/data'

def s2_bands_3yr_7m():

    df = pd.read_csv('/mnt/e/Papers/SpatNet/data/yy_s2_3yr_7m.csv')

    df = df.iloc[:,1:].reset_index(drop=True)
    print(df.head())
    data = df.to_numpy()

    reshaped_data = data.reshape(582, 3, 7, 10)

    print(reshaped_data.shape)
    print("checking: ", reshaped_data[0, :1, :, :])

    output_path = os.path.join(output_dir, "s2_3yr_7m.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(reshaped_data, f)

    print(f"Saved: {output_path}")

    

def k_folds_valid(sample_num=582, k=5):
    data = list(range(sample_num))

    # 5 folds validation
    folds = k
    fold_size = len(data) // folds

    train_test_idx_dir = os.path.join(output_dir, "train_test_idx")

    if not os.path.exists(train_test_idx_dir):
        os.makedirs(train_test_idx_dir)

    for fold in range(folds):
        random.shuffle(data)
        
        test_indices = data[fold * fold_size: (fold + 1) * fold_size]
        train_indices = data[:fold * fold_size] + data[(fold + 1) * fold_size:]

        train_idx_path = os.path.join(train_test_idx_dir, f'train_{fold + 1}.pkl')
        test_idx_path = os.path.join(train_test_idx_dir, f'test_{fold + 1}.pkl')
        
        with open(train_idx_path, 'wb') as f_train:
            pickle.dump(train_indices, f_train)
        
        with open(test_idx_path, 'wb') as f_test:
            pickle.dump(test_indices, f_test)

        print(f"Fold {fold + 1}:")
        print(f"  Training set size: {len(train_indices)}")
        print(f"  Test set size: {len(test_indices)}")
        print(f"  Test indices: {test_indices}")

    


if __name__ == '__main__':
    # s2_bands_3yr_7m()
    k_folds_valid(sample_num=582, k=5)
    
