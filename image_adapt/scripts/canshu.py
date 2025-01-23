import matplotlib.pyplot as plt
import numpy as np

# Set global font to DejaVu Serif
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif'
})

prompt_lengths = [0.1, 0.3, 0.5, 0.7, 0.9]
topic1_values = [59.45,59.69,	59.45,	59.02,	58.96
]
topic5_values = [50.24,	50.34,	50.29,	50.27,	50.31
]
topic8_values = [63.49,	63.59,	63.52,	62.47,	62.68
]

# # 数据准备
# prompt_lengths = [0.1, 1, 10, 100, 1000]
# topic1_values = [58.96, 59.03, 59.38, 59.69, 58.88]
# topic5_values = [49.45, 49.66, 49.79, 50.27, 48.68]
# topic8_values = [63.49, 63.59, 63.52, 62.47, 62.68]

# 创建图表
plt.figure(figsize=(8, 8))

# 绘制三条折线
plt.plot(prompt_lengths, topic1_values, 'red', marker='o', label='gaussian noise')
plt.plot(prompt_lengths, topic5_values, 'blue', marker='x', label='motion blur')
plt.plot(prompt_lengths, topic8_values, 'green', marker='s', label='jpeg compression')
# 设置图表格式
plt.grid(True)


# 设置图表格式
plt.grid(True)
gamma_lower = r'$\gamma$'
plt.xlabel(gamma_lower, fontsize=30, fontfamily='DejaVu Serif')  # 设置横轴标签和字体大小
# eta = r'$\eta$'
# plt.xlabel(eta, fontsize=30, fontfamily='DejaVu Serif')
plt.ylabel('Accuracy (%)', fontsize=30, fontfamily='DejaVu Serif')
plt.title('', fontsize=30, fontfamily='DejaVu Serif')

plt.xlim(0, 1)  # 设置 x 轴范围
plt.xticks(prompt_lengths)  # x 轴刻度为 prompt_lengths 的值，间距相等
# 设置x轴刻度
# plt.xscale('log')
# plt.xticks(prompt_lengths, [str(x) for x in prompt_lengths])
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

# 设置y轴范围
plt.ylim(40, 70)
yticks = np.arange(40, 70.1, 5)
plt.yticks(yticks)

# 添加图例
plt.legend(prop={'family': 'DejaVu Serif', 'size': 20}, loc='best')

# 设置网格线样式为虚线
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图表
plt.savefig("canshu_mi.pdf", format="pdf")

# 显示图表
plt.show()