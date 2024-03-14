import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    '关键参数': ['电流1', '加载油压力', '电机线圈温度3', '给煤量', '电机轴承温度1', '一次风流量', '比例溢流阀开度', '电机轴承温度2', '辊轴承润滑油温度3', '环境温度'],
    '关联度': [0.870111, 0.863444, 0.833234, 0.796272, 0.790431, 0.755219, 0.743810, 0.731221, 0.621829, 0.595534]
}
matplotlib.rcParams['font.family'] = 'SimHei'  # 指定 'SimHei' 字体，该字体支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建DataFrame
df = pd.DataFrame(data)

# 绘制条形图
bars = plt.bar(df['关键参数'], df['关联度'])  # 设置条形图为黑色

# 在每个条形上显示数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3),
             va='bottom', color='black', fontsize=15, fontweight='bold')  # round to 3 decimal places

# 设置标题和轴标签
plt.title('关键参数关联度条形图', color='black', fontsize=20, fontweight='bold')
plt.xlabel('关键参数', fontsize=20, color='black')  # 增加字体大小
plt.ylabel('关联度', fontsize=20, color='black')  # 增加字体大小

# 设置x轴标签旋转角度和字体大小
plt.xticks(fontsize=18, rotation=25)  # 可以根据需要调整字体大小

plt.tight_layout()
# 设置图表风格为黑白
# plt.style.use('grayscale')

# 显示图形
plt.show()
