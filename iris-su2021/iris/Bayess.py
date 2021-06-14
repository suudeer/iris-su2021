import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
plt.rcParams['savefig.dpi'] = 150 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
#贝叶斯分类器
class BayesParameter(): #存储贝叶斯分类器参数

	def __init__(self,mean,cov,category):
		self.mean=mean
		self.cov=cov
		self.category=category
		
class  BayesClassifier():  #贝叶斯分类器,高斯分布概率估计

	def __init__(self):
		self.parameters=[]

	def train(self,X_data,Y_data):

		for categorys in set(Y_data):#遍历每一种类别
			selected= Y_data==categorys #选中对应该类别的数据
			X_newData= X_data[selected] #得到新数据
			mean=np.mean(X_newData,axis=0) #得到均值
			cov = np.cov(X_newData.transpose()) #注意坑 或者设定参数np.cov(X_newData, rowvar=False)
			self.parameters.append(BayesParameter(mean,cov,categorys))

	def predit(self,data):
		res=-1
		probability=0
		for parameter in self.parameters:
			if stats.multivariate_normal.pdf(data, mean=parameter.mean, cov=parameter.cov)>probability:
				res=parameter.category
				probability=stats.multivariate_normal.pdf(data, mean=parameter.mean, cov=parameter.cov)
		return res


def	K_Folds_Cross_Validation(data,tar,k):
	import random
	import numpy as np
	Set=[]
	Tar=[]
	listNum = []
	for i in range(k):
		tempSet=[]
		tempTar=[]
		tempSet.extend(data[i*10:(i+1)*10])
		tempTar.extend(tar[i*10:(i+1)*10])
		tempSet.extend(data[(i+5) * 10:(i + 6) * 10])
		tempTar.extend(tar[(i+5) * 10:(i + 6) * 10])
		tempSet.extend(data[(i+10) * 10:(i + 11) * 10])
		tempTar.extend(tar[(i+10) * 10:(i + 11) * 10])
		Set.append(tempSet)
		Tar.append(tempTar)
	return np.asarray(Set),np.asarray(Tar)

def data_visualization(data,tar):
	trainSet,testSet, trainTar,testTar  = train_test_split(data, tar, test_size=0.3)
	bc = BayesClassifier()
	bc.train(trainSet, trainTar)
	testPredict = np.array([bc.predit(x) for x in testSet],dtype="int")

	import math
	# 画图部分
	fig = plt.figure(figsize=(8, 8))
	xx = [[0, 1], [1, 2], [2, 3], [0,2],[0,3],[1,3]]
	yy = [["sepal_length", "sepal_width"],
		  ["sepal_width", "petal_length"],
		  ["sepal_width", "petal_width"],
		  ["sepal_length","petal_length"],
		  ["sepal_length ","petal_width"],
		  ["sepal_width","petal_width"]]
	for i in range(6):
		ax = fig.add_subplot(321 + i)
		x_max,x_min=testSet.max(axis=0)[xx[i][0]]+0.5,testSet.min(axis=0)[xx[i][0]]-0.5
		y_max,y_min=testSet.max(axis=0)[xx[i][1]]+0.5,testSet.min(axis=0)[xx[i][1]]-0.5
		xlist = np.linspace(x_min, x_max, 80)  
		ylist = np.linspace(y_min, y_max, 100)
		XX, YY = np.meshgrid(xlist, ylist)
		bc = GaussianNB()
		bc.fit(trainSet[:, xx[i]],trainTar)
		xys = [np.array([xx, yy]).reshape(1, -1) for xx, yy in zip(np.ravel(XX), np.ravel(YY))]
		zz = np.array([bc.predict(x) for x in xys])
		Z = zz.reshape(XX.shape)
		plt.contourf(XX, YY, Z, 2, alpha=.1, colors=('blue', 'red', 'green'))
		ax.scatter(testSet[testPredict == 0, xx[i][0]], testSet[testPredict == 0, xx[i][1]],
				  c='r', marker='o',
				   label="setosa")
		ax.scatter(testSet[testPredict==1, xx[i][0]], testSet[testPredict==1, xx[i][1]], c='b', marker='x',
				   label="versicolor")
		ax.scatter(testSet[testPredict==2, xx[i][0]], testSet[testPredict==2, xx[i][1]], c='g', marker='^',
				   label="virginica")
		ax.set_xlabel(yy[i][0])
		ax.set_ylabel(yy[i][1])
		ax.legend(loc=0)
	plt.show()
    
datas=datasets.load_iris()
data=datas.data
tar=datas.target
data_visualization(data,tar)

if __name__=="__main__":
    sets,tar=K_Folds_Cross_Validation(data,tar,5)
    accuracy=0
    print(tar[0].shape)
    for i in range(5): #第i个子集作为测试集
        x,y=0,0
        X_data,Y_data=None,None
        for j in range(5):
            if i!=j:
                if x*y==0:
                    X0_data=sets[i]
                    Y0_data=tar[i]
                else:
                    X0_data=np.concatenate((X0_data,sets[i]),axis=0)
                    Y0_data = np.concatenate((Y0_data, tar[i]), axis=0)
                    x+=1
                    y+=1

        bc= BayesClassifier() 	
        bc.train(X0_data,Y0_data)
		
        y_predict=[bc.predit(x) for x in sets[i]]
        tempAccuracy=np.sum(y_predict==tar[i])/tar[i].shape[0]
        accuracy+=tempAccuracy

    accuracy=accuracy/5
    print("accuracy:",accuracy)