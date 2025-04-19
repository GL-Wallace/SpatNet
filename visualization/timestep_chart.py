import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap



def plot_data(df):

    x_labels = df.index
    x_positions = range(len(x_labels))
    plt.figure(figsize=(12, 8))
    for col in df.columns:
        plt.plot(x_positions,df[col],marker='x', label=col)

    plt.xticks(x_positions,x_labels, rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Time span from 2019 to 2021')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    y_min = df.values.min()
    y_max = df.values.max()
    plt.ylim(bottom=y_min, top=y_max)

    plt.show()

def format_data(path):
    df = pd.read_csv(path)
    columns = df.columns[1:]

    column_groups = {}
    for col in columns:
        if '_' in col:  
            base, year = col.split('_')
            if base not in column_groups:
                column_groups[base] = []
            column_groups[base].append((int(year), col))

    sorted_columns = []
    for b_group in column_groups:
        sorted_columns.extend([col for _, col in sorted(column_groups[b_group], key=lambda x: x[0])])

    df_sorted = df[sorted_columns]

    columns = df_sorted.columns

    years = sorted(set(int(col.split('_')[1]) for col in columns if '_' in col))
    bands = sorted(set(col.split('_')[0] for col in columns if '_' in col))
    bands = sorted(bands, key=lambda x: (int(re.sub(r'\D', '', x[1:])), x))

    # print("Years:", years)
    # print("Bands:", bands)

    data_dict = {year: [] for year in years}

    for col in columns:
        if '_' in col:
            band, year = col.split('_')
            data_dict[int(year)].append(df_sorted[col].values[0])  # 使用 df_sorted

    # print("Data dict by year:")
    # print(data_dict)

    df_t = pd.DataFrame(data_dict)
    df_t = df_t.astype(int)
    df_transposed = df_t.transpose()
    return df_transposed


def plot_all_data(df1, df2, band_columns):
    # 设置图形大小
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # 计算所需的颜色数量
    total_colors_needed = len(band_columns)
    # 设置亮色系配色方案，确保有足够的颜色
    palette = sns.color_palette("bright", n_colors=total_colors_needed)
    
    # # 自定义颜色映射，以 lightgreen 为基础
    # colors = ['white', 'lightgreen']
    # cmap = LinearSegmentedColormap.from_list('lightgreen_cmap', colors)
    # palette = [cmap(i / total_colors_needed) for i in range(total_colors_needed)]
    # 第一个子图的绘图设置
    # 作为横坐标的标签
    x_labels1 = df1.index
    # 横坐标的位置
    x_positions1 = range(len(x_labels1))

    # 绘制第一个子图的折线
    for i, col in enumerate(df1.columns):
        ax1.plot(x_positions1, df1[col], marker='x', label=band_columns[i], color=palette[i], linewidth=2)

    # 设置第一个子图的坐标轴刻度
    ax1.set_xticks(x_positions1)
    ax1.set_xticklabels(x_labels1, rotation=45, fontsize=15)
    # 设置第一个子图的标题
    ax1.set_title('Time span from 2019 to 2021 (Apr. - Oct.)', fontsize=18, fontweight='bold')
    # 设置第一个子图的网格
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 获取第一个子图数据的最小值和最大值
    y_min1 = df1.values.min()
    y_max1 = df1.values.max()
    # 设置第一个子图的y轴范围
    ax1.set_ylim(bottom=y_min1, top=y_max1)

    # 第二个子图的绘图设置
    # 作为横坐标的标签
    x_labels2 = df2.index
    # 横坐标的位置
    x_positions2 = range(len(x_labels2))

    # 绘制第二个子图的折线
    for i, col in enumerate(df2.columns):
        ax2.plot(x_positions2, df2[col], marker='x', color=palette[i], linewidth=2)

    # 设置第二个子图的坐标轴刻度
    ax2.set_xticks(x_positions2)
    ax2.set_xticklabels(x_labels2, rotation=45, fontsize=16)
    # 设置第二个子图的标题
    ax2.set_title('Time span from 2019 to 2021 (Jan. - Dec.)', fontsize=18, fontweight='bold')
    # 设置第二个子图的网格
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 获取第二个子图数据的最小值和最大值
    y_min2 = df2.values.min()
    y_max2 = df2.values.max()
    # 设置第二个子图的y轴范围
    ax2.set_ylim(bottom=y_min2, top=y_max2)

    # 获取第一个子图的图例句柄和标签
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', fontsize=14, ncol=len(band_columns))

    # 调整子图布局，增加第一个子图上方的空间
    plt.subplots_adjust(top=0.9, hspace=0.4)
    # plt.tight_layout()
    save_path = './visualization/timelength.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


df1 = format_data('/mnt/e/Papers/SpatNet/data/yy_s2_3yr_7m.csv')
df2 = format_data('/mnt/e/Papers/SpatNet/data/yy_s2_3yr_12m.csv')
band_columns = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
plot_all_data(df1, df2, band_columns)