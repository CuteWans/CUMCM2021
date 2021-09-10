import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"/usr/share/fonts/consolas-with-yahei/consnerdi.ttf", size=12)
x1 = [250, 275, 300, 325, 350]
workbook = xlrd.open_workbook(r'wt.xlsx')
sheet = workbook.sheet_by_index(0)
x2 = [0.5, 1, 2, 5]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ys = np.array([0.5, 1, 2, 5])
for i in range(0,5):
    z = sheet.row_values(i)
    zs = np.array(z)
    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
    ax.plot(ys, zs, x1[i], zdir='y', color='b', marker='o', alpha=0.8, label = str(x1[i]) + '℃')

x1 = np.array(x1)
for i in range(0,4):
    z = sheet.col_values(i)
    x = x2[i]
    zs = np.array(z)
    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
    ax.plot(x1, zs, x, zdir='x', color='b', marker='o', alpha=0.8, label = str(x2[i]) + 'wt%')

plt.title('Co负载量与乙醇转化率关系图', fontproperties = font)
ax.set_xlabel('Co负载量', fontproperties = font)
ax.set_zlabel('乙醇转化率', fontproperties = font)
ax.set_ylabel('温度', fontproperties = font)
#ax.legend()
plt.savefig("温度&wt.png")
plt.show()
