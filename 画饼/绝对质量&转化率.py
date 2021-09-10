#A1 A2 A4 A6
import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"/usr/share/fonts/consolas-with-yahei/consnerdi.ttf", size=12)
workbook = xlrd.open_workbook(r'绝对质量&转化率.xlsx')
sheet = workbook.sheet_by_index(0)
x1 = [10, 25, 50, 75, 100]
x2 = [250, 275, 300, 350, 400]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

for i in range(0, 5):
    z = sheet.col_values(i)
    x = x1[i]
    xs = np.array(x2)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='x', color='r', marker='o', alpha=0.8)

for i in range(0, 5):
    z = sheet.row_values(i)
    x = x2[i]
    xs = np.array(x1)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='y', color='r', marker='o', alpha=0.8)

plt.title('绝对质量与乙醇转化率关系图', fontproperties = font)
ax.set_xlabel('绝对质量', fontproperties = font)
ax.set_zlabel('乙醇转化率', fontproperties = font)
ax.set_ylabel('温度', fontproperties = font)
#ax.legend(loc = 'upper left')
plt.savefig("绝对质量&转化率.png")
plt.show()