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
from collections import defaultdict
import csv

output_dir = '/mnt/e/PythonPractice/SpatNet/data'

def s2_bands2pkl(file_loc, samples, year, month, bands):

    df = pd.read_csv(file_loc)

    df = df.iloc[:,2:].reset_index(drop=True)
    print(df.head())
    data = df.to_numpy()

    reshaped_data = data.reshape(samples, year, month, bands)

    print(reshaped_data.shape)
    output_name = f's2_{year}yr_{month}m_{bands}bs.pkl'
    output_path = os.path.join(output_dir, output_name)
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


def s2_gee_samples():
    df = pd.read_csv('/mnt/e/Papers/SpatNet/data/yy_s2_3yr_12m_org.csv')
    formated_csv_save_path = '/mnt/e/Papers/SpatNet/data/yy_s2_3yr_12m.csv'
    unique_ids = df['IDID'].drop_duplicates()
    print('Number of Unique Ids: ', len(unique_ids))

    som_dict = {}
    band_columns = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    for unique_id in unique_ids:
        rows = df[df['IDID'] == unique_id]

        for _, row in rows.iterrows():
            date = row['date']
            year, month = date.split('-')[:2]
            
            band_values = [row[band] for band in band_columns]
            
            if unique_id not in som_dict:
                som_dict[unique_id] = {}

            som = rows['SOM'].iloc[0]
            
            if som not in som_dict[unique_id]:
                som_dict[unique_id][som] = {}

            if year not in som_dict[unique_id][som]:
                som_dict[unique_id][som][year] = {}

            som_dict[unique_id][som][year][month] = band_values


    for unique_id in som_dict:
        for som in som_dict[unique_id]:
            years = sorted(som_dict[unique_id][som].keys())
            for year_index, year in enumerate(years):
                months = sorted(som_dict[unique_id][som][year].keys())
                all_months = set(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
                missing_months = all_months - set(months)
                for month_index, missing_month in enumerate(missing_months):

                    if year_index + 1 < len(years):
                        next_year = years[year_index + 1]
                        result = som_dict[unique_id][som][year].setdefault(str(missing_month).zfill(2), som_dict[unique_id][som][next_year][str(missing_month).zfill(2)])
                        if result is None or result == []:
                            next_year_next = years[year_index + 2]
                            som_dict[unique_id][som][year].setdefault(str(missing_month).zfill(2), som_dict[unique_id][som][next_year_next][str(missing_month).zfill(2)])

                    if year_index > 0: 
                        prev_year = years[year_index - 1]
                        result = som_dict[unique_id][som][year].setdefault(str(missing_month).zfill(2), som_dict[unique_id][som][prev_year][str(missing_month).zfill(2)])
                        if result is None or result == []:
                            year_prev_year = years[year_index - 2]
                            som_dict[unique_id][som][year].setdefault(str(missing_month).zfill(2), som_dict[unique_id][som][year_prev_year][str(missing_month).zfill(2)])

    # for unique_id in som_dict:
    #     for som in som_dict[unique_id]:
    #         years = sorted(som_dict[unique_id][som].keys())
            
    #         for year in years:
    #             months = sorted(som_dict[unique_id][som][year].keys())
    #             all_months = set(str(i).zfill(2) for i in range(1, 13))  
                
    #             missing_months = all_months - set(months)  
                
    #             if missing_months:
    #                 print(f"Data is missing for {unique_id} in {som} for year {year}. Missing months: {missing_months}")
    #             else:
    #                 print(f"All months are complete for {unique_id} in {som} for year {year}.")


    rows = []

    for som_id, som_data in som_dict.items():
        for som_value, year_data in som_data.items():
            row = [som_id, som_value]  
            years = sorted(year_data.keys())
            for year in years:
                months = sorted(year_data[year].keys())
                for month in months:
                    for band in band_columns:
                        column_name = f"{year}-{month}-{band}"
                        band_data = year_data[year][month][band_columns.index(band)]
                        row.append(band_data)
            rows.append(row)

    headers = ['id', 'som']  
    i=0

    for year in years:
        months = sorted(som_dict[unique_id][som][year].keys())
        for month in months:
            for band in band_columns:
                column_name = f"{band}_{year}{month}"
                i=i+1
                headers.append(column_name)

    with open(formated_csv_save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  
        # sorted_rows = sorted(rows, key=lambda row: int(row['som']))
        sorted_rows = sorted(rows, key=lambda column:column[1])
        writer.writerows(sorted_rows) 
            
    print("formating the S2 date finished.")

    s2_bands2pkl(formated_csv_save_path, 582, 3, 12, 10)
    print("DONE")


def spatial_properities():
    file_loc = '/mnt/e/Papers/SpatNet/data/spa_data.csv'
    df = pd.read_csv(file_loc)
    df = df.iloc[:,1:].reset_index(drop=True)
    print(df.head())
    data = df.to_numpy()

    reshaped_data = data.reshape(582, 17, 1)

    print(reshaped_data.shape)
    output_name = f'spa_data.pkl'
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(reshaped_data, f)
    print(f"Saved: {output_path}")

def spatial_windows():
    file_loc = '/mnt/e/Papers/SpatNet/data/spa_data.csv'
    df = pd.read_csv(file_loc)
    df = df.iloc[:,1:].reset_index(drop=True)
    print(df.head())
    data = df.to_numpy()

    reshaped_data = data.reshape(582, 17, 1)
    expanded_data = np.tile(reshaped_data[:, :, np.newaxis, :], (1, 1, 5, 5))

    print(expanded_data.shape)
    output_name = f'spa_windows_data.pkl'
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(expanded_data, f)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    # k_folds_valid(sample_num=582, k=5)
    # s2_gee_samples()
    # spatial_properities()
    spatial_windows()
    
