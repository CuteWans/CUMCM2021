#A1 A2 A4 A6
import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"/usr/share/fonts/consolas-with-yahei/consnerdi.ttf", size=12)
workbook = xlrd.open_workbook(r'AB&转化率.xlsx')
sheet = workbook.sheet_by_index(0)
x1 = [1.68, 2.1]
x2 = [250, 275, 300, 350, 400]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

for i in range(0, 2):
    z = sheet.col_values(i)
    x = x1[i]
    xs = np.array(x2)
    zs = np.array(z)
    if i == 0:
        ax.plot(xs, zs, x, zdir='x', color='b', marker='o', alpha=0.8)
    else:
        l1 = ax.plot(xs, zs, x, zdir='x', color='b', marker='o', alpha=0.8, label = 'A装填')

for i in range(2, 4):
    z = sheet.col_values(i)
    x = x1[i - 2]
    xs = np.array(x2)
    zs = np.array(z)
    if i == 2:
        ax.plot(xs, zs, x, zdir='x', color='r', marker='o', alpha=0.8)
    else:
        l2 = ax.plot(xs, zs, x, zdir='x', color='r', marker='o', alpha=0.8, label='B装填')

plt.title('装填方式与乙醇转化率关系图', fontproperties = font)
ax.set_xlabel('乙醇浓度', fontproperties = font)
ax.set_zlabel('乙醇转化率', fontproperties = font)
ax.set_ylabel('温度', fontproperties = font)
plt.legend(loc = 'upper left', prop = font)
#ax.legend(loc = 'upper left')
plt.savefig("AB&转化率.png")
plt.show()