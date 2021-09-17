#A1 A2 A4 A6
import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"consnerdi.ttf", size=12)
workbook = xlrd.open_workbook(r'装料比&选择性.xlsx')
sheet = workbook.sheet_by_index(0)
x1 = [250, 275, 300, 350, 400]
x2 = [0, 0.5556, 1, 2.0303]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

for i in range(0, 5):
    z = sheet.row_values(i)
    x = x1[i]
    xs = np.array(x2)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='y', color='b', marker='o', alpha=0.8)

for i in range(0, 4):
    z = sheet.col_values(i)
    x = x2[i]
    xs = np.array(x1)
    zs = np.array(z)
    ax.plot(xs, zs, x, zdir='x', color='b', marker='o', alpha=0.8)

plt.title('HAP和Co/SiO2装料比与C4烯烃选择性关系图', fontproperties = font)
ax.set_xlabel('HAP和Co/SiO2装料比', fontproperties = font)
ax.set_zlabel('C4烯烃选择性', fontproperties = font)
ax.set_ylabel('温度', fontproperties = font)
plt.savefig("装料比2.png")
plt.show()