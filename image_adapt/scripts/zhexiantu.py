import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Set the global font to Times New Roman
plt.rcParams['font.family'] = 'DejaVu Serif'

# 数据
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
tent_random_order = [36.1, 44, 45.2, 46.4, 47.6, 48.8, 50.1]  # 示例数据
tent_class_order = [43.41, 46.23, 48.78, 50.65, 50.79, 50.89, 50.92]
diffpure = [28.8] * len(batch_sizes)
memo = [37.8] * len(batch_sizes)
dda = [44.2] * len(batch_sizes)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, tent_class_order, '-p', color='orange', label='CMDA, random order')
plt.plot(batch_sizes, tent_random_order, '-.o', label='Tent, random order')
plt.plot(batch_sizes, diffpure, ':s', label='DiffPure: 28.8%')
plt.plot(batch_sizes, memo, '--x', label='MEMO: 37.8%')
plt.plot(batch_sizes, dda, '-o', label='DDA: 44.2%')

# 设置图例
font = FontProperties(family='DejaVu Serif', size=18)
plt.legend(prop=font, loc='upper left')

# 设置轴标签
plt.xlabel('Batch Size', fontsize=18, fontfamily='DejaVu Serif')
plt.ylabel('Accuracy (%)', fontsize=18, fontfamily='DejaVu Serif')

# 修改坐标轴刻度字体大小和字体
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)

# 添加图标题和标注
plt.title('ConvNeXt-Tiny', fontsize=18, fontfamily='DejaVu Serif')

# 设置坐标刻度
plt.xscale('log', base=2)
plt.xticks(batch_sizes, batch_sizes)
plt.yticks(np.arange(20, 80, 10))
plt.ylim(20, 80)

# 显示图形
plt.tight_layout()
plt.savefig("zhexiantu_CMDA.pdf", format="pdf")
plt.show()