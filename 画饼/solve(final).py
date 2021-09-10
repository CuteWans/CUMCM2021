import os, re
import numpy as np
import pandas as pd

train = pd.read_csv('./data-set/train.csv')
test = pd.read_csv('./data-set/test.csv')

#查看清洗前数据
print('清洗前：')
print('训练数据',train['text'].head())
print('测试数据',test['text'].head())

#普适清洗
def clean_text(text):
	temp = text.lower()                                 #文档转换为小写
	temp = re.sub('\n', ' ', temp)                      #删除换行符
	temp = re.sub('\'', '', temp)                       #删除引号
	temp = re.sub('-', ' ', temp)                       #删除‘-’
	temp = re.sub(r'(http|https|pic.)\S', ' ', temp)    #删除网址及引用图片
	temp = re.sub(r'[^\w\s]', ' ', temp)                #删除可见及不可见符号
	
	return temp

#清洗虚词
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_functionwords(text):
	temp = ' '.join([text for text in text.split() if len(text) > 3])      #删除过短的单词
	tokenized_words = word_tokenize(temp)
	stop_words = set(stopwords.words('english'))                           #获取英语停止词
	temp = [word for word in tokenized_words if word not in stop_words]    #删除停止词
	temp = ' '.join(temp)
	
	return temp

#清洗训练数据
train['clean'] = train['text'].apply(clean_text)
train['clean'] = train['clean'].apply(remove_functionwords)

#清洗测试数据
test['clean'] = test['text'].apply(clean_text)
test['clean'] = test['clean'].apply(remove_functionwords)

#查看清洗后数据
print('清洗后：')
print('训练数据', train['clean'].head())
print('测试数据', test['clean'].head())

# 词云
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text0 = []; text1 = [];
cnt0 = 0; cnt1 = 0;
for i in range(0, 5329):
	if train['target'][i] == 0:
		text0.append(train['clean'][i])
		cnt0 += 1
	else:
		text1.append(train['clean'][i])
		cnt1 += 1
print('训练数据中 False : True = %d : %d' %(cnt0, cnt1))

# for not disaster tweets
plt.figure(figsize = (15, 8))
word_cloud = WordCloud(background_color = 'white', max_font_size = 80).generate(" ".join(text0))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

# for disaster tweets
plt.figure(figsize = (15, 8))
word_cloud = WordCloud(background_color = 'white', max_font_size = 80).generate(" ".join(text1))
plt.imshow(word_cloud)
plt.axis('off')	
plt.show()

#将文本与关键词进行结合
def combine_attributes(text, keyword):
	temp = [text, keyword]
	combined = ' '.join(x for x in temp if x)
	return combined

train.fillna('', inplace = True) #将缺失信息填充为空串
train['combine'] = train.apply(lambda x: combine_attributes(x['clean'],x['keyword']), axis = 1)

test.fillna('', inplace = True)
test['combine'] = test.apply(lambda x: combine_attributes(x['clean'],x['keyword']), axis = 1)

#If-IDF预处理
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_vect_all = vectorizer.fit_transform(train['combine'])
X_test_vect_all = vectorizer.transform(test['combine'])

#十折交叉验证法
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

KF = KFold(n_splits = 10, shuffle = True)
X = train['combine']
y = train['target']

test_num = 0
ans_pre_all = []
Accuracy_score = 0
Precision_score = 0
Recall_score = 0
F1_score = 0
for i in range(0, 2284):
	ans_pre_all.append(0)
print('')
for train_index, test_index in KF.split(X):
	test_num += 1
	print('# 第%d次训练开始：' %(test_num))
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	print ('原始数据集特征：',X.shape, 
		   '训练数据集特征：',X_train.shape,
		  '测试数据集特征：',X_test.shape)

	print ('原始数据集标签：',y.shape, 
		   '训练数据集标签：',y_train.shape,
	  	  '测试数据集标签：',y_test.shape)
	
	#Tf-IDF加权
	X_train_vect = vectorizer.transform(X_train)
	X_test_vect = vectorizer.transform(X_test)
	
	#支持向量机SVC进行学习
	clf = SVC(kernel = 'linear')
	clf.fit(X_train_vect, y_train)
	
	y_pred = clf.predict(X_test_vect)

	#评估模型准确度
	temp = accuracy_score(y_test, y_pred)
	Accuracy_score += temp
	print('准确率：', temp)
	
	temp = precision_score(y_test, y_pred, average = 'binary')
	Precision_score += temp
	print('精确率：', temp)
	
	temp = recall_score(y_test, y_pred, average = 'binary')
	Recall_score += temp
	print('召回率：', temp)
	
	temp = f1_score(y_test, y_pred, average = 'binary')
	F1_score += temp
	print('F1分数：', temp)
	
	print('本次训练结束\n')
	
	y_pred_all = clf.predict(X_test_vect_all)
	ans_pre_all += y_pred_all

#输出结果
print('总准确率：', Accuracy_score / 10);
print('总精确率：', Precision_score / 10);
print('总召回率：', Recall_score / 10);
print('总F1分数：', F1_score / 10);

for i in range(0, 2284):
	ans_pre_all[i] /= 10
	ans_pre_all[i] = round(ans_pre_all[i])
np.savetxt('./data-set/result.csv', y_pred_all, fmt = '%d', delimiter = ',')