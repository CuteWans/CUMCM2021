#A1 A2 A4 A6
import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"/usr/share/fonts/consolas-with-yahei/consnerdi.ttf", size=12)
workbook = xlrd.open_workbook(r'wt.xlsx')
sheet = workbook.sheet_by_index(0)
x1 = [0.5, 1, 2, 5]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
for i in range(0,8,2):
    print(i)
    x2 = sheet.col_values(i)
    z = sheet.col_values(i + 1)
    x = x1[i // 2]
    print(x)
    zs = np.array(z)
    print(zs)
    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
    ax.plot(x2, zs, x, zdir='x', color=color, marker='o', alpha=0.8, label = str(x1[i//2]) + 'wt%')
plt.title('温度与乙醇转化率关系图', fontproperties = font)
ax.set_xlabel('Co负载量', fontproperties = font)
ax.set_zlabel('乙醇转化率', fontproperties = font)
ax.set_ylabel('温度', fontproperties = font)
ax.legend()
plt.savefig("温度1.png")
plt.show()
