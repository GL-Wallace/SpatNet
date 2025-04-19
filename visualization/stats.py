import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split, StratifiedKFold

sns.set(style="ticks", palette="deep")

som_data = pd.read_csv('./data/sample_values.csv') 

som_data = som_data.iloc[:, 0]

A_group, B_group = train_test_split(som_data, test_size=0.1, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

num_bins = 10
A_group_array = A_group.values.flatten()
bins = np.linspace(np.min(A_group_array), np.max(A_group_array), num_bins + 1)
labels = np.digitize(A_group_array, bins)

train_data = []
valid_data = []

for fold_num, (train_index, test_index) in enumerate(skf.split(A_group_array, labels), start=1):
    if fold_num == 5:
        valid_data = A_group_array[test_index]
    else:
        train_data.extend(A_group_array[train_index])

train_data = np.array(train_data)

test_data = B_group.values.flatten()

mu, std = norm.fit(som_data)

plt.figure(figsize=(12, 6))
plt.rcParams['font.family'] = 'Times New Roman'

# 使用明亮色系绘制直方图
sns.histplot(som_data, bins=30, kde=False, color='lightskyblue', label='Total Data', edgecolor='black', alpha=0.6)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p * len(som_data) * (x[1] - x[0]), 'r', label='Gaussian Fit', linewidth=2)

sns.histplot(train_data, bins=20, kde=False, color='lightgreen', label='Train Data', edgecolor='black', alpha=0.4)
sns.histplot(valid_data, bins=20, kde=False, color='yellow', label='Validation Data', edgecolor='black', alpha=0.6)
sns.histplot(test_data, bins=20, kde=False, color='orange', label='Test Data', edgecolor='black', alpha=0.6)

plt.title('Samples Distribution Analysis', fontsize=20, fontweight='bold')
plt.xlabel('SOM Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.gca().spines['top'].set_linewidth(1)
plt.gca().spines['right'].set_linewidth(1)
plt.gca().spines['left'].set_linewidth(1)
plt.gca().spines['bottom'].set_linewidth(1)

plt.tick_params(axis='both', which='major', labelsize=14)  

plt.legend(title='Data Sets', title_fontsize='14', fontsize='14', loc='upper right')

plt.tight_layout()
plt.savefig('./visualization/data_distribution.png', dpi=300, bbox_inches='tight')

plt.show()