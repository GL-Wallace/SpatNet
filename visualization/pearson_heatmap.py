import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


df = pd.read_csv('/mnt/e/Papers/SpatNet/data/spa_data.csv')  # Replace with your actual file path

corr_with_target = df.corr(method='pearson')['SOM'].drop('SOM')
corr_df = corr_with_target.to_frame()

plt.figure(figsize=(2, len(corr_df) * 0.5 + 1))
plt.rcParams['font.family'] = 'Times New Roman'

# 设置数值标注的字体大小
annot_font_size = 14
# 设置坐标轴标签的字体大小
axis_label_font_size = 14
# 设置标题的字体大小
title_font_size = 20
# 设置颜色条刻度的字体大小
cbar_tick_font_size = 12
# 自定义颜色映射，以 lightgreen 为基础
colors = ['white', 'lightgreen']
cmap = LinearSegmentedColormap.from_list('lightgreen_cmap', colors)
ax = sns.heatmap(
    corr_df,
    annot=True,  # Show correlation values
    annot_kws={'size': annot_font_size},  # 设置数值标注的字体大小
    cmap=cmap,  # Blue to red gradient
    center=0,  # Centered at 0 correlation
    linewidths=1,  # Thin border lines
    fmt=".2f",  # Format float with 2 decimals
    cbar_kws={'shrink': 0.4},  # Shrink colorbar width
    square=False,
)

# plt.title('Pearson Correlation with SOM', fontsize=title_font_size, fontweight='bold', pad=10)

plt.xticks(rotation=0, fontsize=axis_label_font_size)
plt.yticks(rotation=0, fontsize=axis_label_font_size)

# 设置颜色条刻度的字体大小
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=cbar_tick_font_size)

plt.tight_layout()
save_path = './visualization/heatmap.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()