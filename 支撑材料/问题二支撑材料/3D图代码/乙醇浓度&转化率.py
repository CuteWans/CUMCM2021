#A1 A2 A4 A6
import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"consnerdi.ttf", size=12)
workbook = xlrd.open_workbook(r'乙醇浓度&转化率.xlsx')
sheet = workbook.sheet_by_index(0)
x1 = [250, 275, 300, 350, 400]
x2 = [0.3, 0.9, 1.68, 2.1]
x2_ = [1.68, 2.1]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

for i in range(0, 5):
    z = sheet.row_values(i)
    x = x1[i]
    xs = np.array(x2)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='y', color='b', marker='o', alpha=0.8)

workbook_ = xlrd.open_workbook(r'乙醇B.xlsx')
sheet_ = workbook_.sheet_by_index(0)
for i in range(0, 5):
    z = sheet_.row_values(i)
    x = x1[i]
    xs = np.array(x2_)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='y', color='r', marker='o', alpha=0.8)

l1 = []
l2 = []
for i in range(0, 3):
    z = sheet.col_values(i)
    x = x2[i]
    xs = np.array(x1)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='x', color='b', marker='o', alpha=0.8)

z = sheet.col_values(3)
x = x2[3]
xs = np.array(x1)
zs = np.array(z)
ax.plot(xs, zs, x, zdir='x', color='b', marker='o', alpha=0.8, label = 'A装填方式')


for i in range(0, 1):
    z = sheet_.col_values(i)
    x = x2_[i]
    xs = np.array(x1)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='x', color='r', marker='o', alpha=0.8)

z = sheet_.col_values(1)
x = x2_[1]
xs = np.array(x1)
zs = np.array(z)
ax.plot(xs, zs, x, zdir='x', color='r', marker='o', alpha=0.8, label = 'B装填方式')

plt.title('乙醇浓度与乙醇转化率关系图', fontproperties = font)
ax.set_xlabel('乙醇浓度', fontproperties = font)
ax.set_zlabel('乙醇转化率', fontproperties = font)
ax.set_ylabel('温度', fontproperties = font)
plt.legend(loc = 'upper left', prop = font)
#ax.legend(loc = 'upper left')
plt.savefig("浓度1.png")
plt.show()