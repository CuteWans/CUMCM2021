import random
import matplotlib.pyplot as plt
import numpy as np
import pylab as mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = list(range(0, 24))  # 数据在x轴上的坐标

for z in range(3):  # 这里我们设置z=0到2，代表周一到周三
    ys = np.random.rand(24) * 100  # 数据在y轴上的坐标
    print(ys)
    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))  # 得到一个随机的颜色用于下面绘制该条折线图
    ax.plot(xs, ys, zs=z, zdir='y', color=color, marker='o', alpha=0.8)

# 在设置zdir = 'y'的情形下，其实y轴才是z轴，然后z轴变成了y轴
ax.set_xlabel('X轴')
ax.set_ylabel('Z轴')
ax.set_zlabel('Y轴')

plt.show()