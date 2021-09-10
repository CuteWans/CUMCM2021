import xlrd
import random
import pylab as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"/usr/share/fonts/consolas-with-yahei/consnerdi.ttf", size=12)
workbook = xlrd.open_workbook(r'wt2.xlsx')
sheet = workbook.sheet_by_index(0)
x1 = [250, 275, 300, 325, 350]
print(x1)
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ys = np.array([0.5, 1, 2, 5])
p = []
for i in range(0,5):
    print(x1[i])
    z = sheet.row_values(i)
    zs = np.array(z)
    print(ys)
    print(zs)
    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
    ax.plot(ys, zs, x1[i], zdir='y', color=color, marker='o', alpha=0.8, label = str(x1[i]) + '℃')
print(p)
plt.title('Co负载量与C4烯烃选择性关系图', fontproperties = font)
ax.set_xlabel('Co负载量', fontproperties = font)
ax.set_zlabel('C4烯烃选择性', fontproperties = font)
ax.set_ylabel('温度', fontproperties = font)
ax.legend()
plt.savefig("wt2.png")
plt.show()
