#A1 A2 A4 A6
import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"consnerdi.ttf", size=12)
workbook = xlrd.open_workbook(r'A二维图.xlsx')
sheet = workbook.sheet_by_index(0)
x1 = [250, 275, 300, 350, 400]
x2 = [250, 275, 300, 325, 350]

fig = plt.figure()
ax = fig.add_subplot()
z1 = sheet.col_values(2)
xs = np.array(x1)
zs = np.array(z1)
ax.plot(xs, zs, color='r', marker='o', alpha=0.8, label = '50mg')

z2 = sheet.col_values(3)
xs = np.array(x2)
zs = np.array(z2)
ax.plot(xs, zs, color='b', marker='o', alpha=0.8, label = '200mg')

plt.title('绝对质量&C4烯烃选择性关系图', fontproperties = font)
ax.set_ylabel('C4烯烃选择性', fontproperties = font)
ax.set_xlabel('温度', fontproperties = font)
ax.legend()
plt.savefig("A绝对质量&选择性.png")
plt.show()