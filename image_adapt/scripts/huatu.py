import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['gaussian', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom',
              'frost', 'snow', 'fog', 'bright', 'contrast', 'elastic', 'pixel', 'jpeg']
# source = [30.25, 28.40, 28.36, 21.61, 9.97, 24.45, 27.31, 35.76, 47.15, 48.99, 67.31, 34.66, 21.89, 31.71, 50.76, ]
#
# dda = [51.21, 46.34, 46.06, 20.83, 21.90, 23.69, 27.65, 36.04, 45.25, 40.22, 63.98, 30.42, 40.20, 48.29, 54.01, ]
#
# sda = [52.37, 50.24, 50.09, 24.43, 26.47, 29.09, 32.36, 37.26, 39.43, 38.83, 65.29, 26.11, 46.05, 55.28, 57.21, ]
#
# our = [55.76, 50.96, 50.27, 35.17, 36.23, 39.55, 40.14, 35.63, 47.15, 47.42, 67.30, 32.55, 50.34, 56.93, 58.24, ]
source = [40.15, 39.12, 38.74, 25.24, 11.38, 33.29, 31.34, 43.76, 49.21, 42.34, 70.36, 43.80, 22.58, 37.36, 57.22, ]

dda = [55.57, 51.57, 51.27, 24.65, 26.86, 31.87, 32.27, 42.60, 48.40, 34.26, 66.65, 39.86, 42.18, 54.60, 59.31, ]

sda = [56.71, 54.02, 53.60, 29.94, 31.73, 35.89, 36.83, 44.07, 49.83, 38.79, 68.51, 38.39, 47.05, 59.72, 62.01,]

our = [59.69, 55.39, 54.83, 44.21, 44.84, 49.35, 46.40, 44.47, 49.21, 42.32, 70.36, 43.82, 53.06, 62.51, 63.24, ]

# 调整每列柱形图之间的间隙
gap = 0.02  # 稍微增加间隙值
x = np.arange(len(categories))  # 横坐标位置
width = 0.2  # 柱子的宽度

# 创建图形
fig, ax = plt.subplots(figsize=(18, 4))

# 绘制柱状图，每个柱子之间加入间隙
ax.bar(x - 1.5*(width + gap), source, width, label='Source:39.0%', color='#1f77b4')
ax.bar(x - 0.5*(width + gap), dda, width, label='DDA:44.1%', color='#ff7f0e')
ax.bar(x + 0.5*(width + gap), sda, width, label='SDA:47.1%', color='#2ca02c')
ax.bar(x + 1.5*(width + gap), our, width, label='CMDA:52.2%', color='#d62728')

# 添加细节
ax.set_ylabel('Accuracy', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, fontsize=14)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_ylim(0, 90)
ax.tick_params(axis='y', labelsize=16)
# 去掉左边纵轴的刻度线
ax.tick_params(left=False)

# 添加图例，并放入框内顶部
legend = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 0.95),  # 图例位置调整到框内顶部
    ncol=4,  # 图例列数
    fontsize=16,
    frameon=True
)
legend.get_frame().set_edgecolor('black')  # 设置图例边框颜色
legend.get_frame().set_linewidth(0.8)      # 设置边框线宽

# 设置紧凑布局
plt.tight_layout()

plt.savefig("zhuxingtu_convnext.pdf", format="pdf")

# 展示图形
plt.show()
