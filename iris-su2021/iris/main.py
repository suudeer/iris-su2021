import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# MED分类器
class Medclass:
    def __init__(self):
        self.center_dict = {}  # 分类中心点，以类别标签为键   label: center_point(list)
        self.feature_number = 0  # 特征维度
        self.train_state = False  # 训练状态，True为训练完成，False表示还没训练过

    def train(self, feature_set, label_set):
        new_label_set = {key: value for key, value in enumerate(label_set)}  # 将标签集合转换为以下标为键的字典   index: label
        self.feature_number = len(feature_set[0])
        sample_num = len(label_set)  # 样本个数
        count = {}  # 计算每个类别的样本个数  label: count(int)
        # 计算每个类别的分类中心点
        for index in range(sample_num):
            if new_label_set[index] not in count.keys():
                count[new_label_set[index]] = 0
            else:
                count[new_label_set[index]] += 1  # 计算对应标签的样本数
            if new_label_set[index] not in self.center_dict.keys():
                self.center_dict[new_label_set[index]] = feature_set[index]
            else:
                self.center_dict[new_label_set[index]] += feature_set[index]
        for _key_ in self.center_dict.keys():
            for _feature_ in range(self.feature_number):
                self.center_dict[_key_][_feature_] /= count[_key_]
        self.train_state = True

    # 根据输入来进行分类预测，输出以 下标—预测分类 为键值对的字典
    def predict(self, feature_set):
        # 先判断此分类器是否经过训练
        if not self.train_state:
            return {}
        sample_num = len(feature_set)
        distance_to = {}  # 计算某个样本到各分类中心点距离的平方  label: float
        result = {}  # 保存分类结果  index: label
        for _sample_ in range(sample_num):
            for _key_ in self.center_dict.keys():
                delta = feature_set[_sample_] - self.center_dict[_key_]
                distance_to[_key_] = np.dot(delta.T, delta)
            result[_sample_] = min(distance_to, key=distance_to.get)  # 返回最小值的键（即label）
        return result

    # 判断预测准确率
    def accuracy(self, feature_set, label_set):
        if not self.train_state:
            return 0.0
        correct_num = 0
        total_num = len(label_set)
        predict = self.predict(feature_set)
        for _sample_ in range(total_num):
            if predict[_sample_] == label_set[_sample_]:
                correct_num += 1
        return correct_num / total_num

    # 根据指定的阳性类别，计算分类器的性能指标（准确率accuracy，精度precision，召回率recall，特异性specificity，F1_Score）
    def performance(self, feature_set, label_set, positive):
        if not self.train_state:
            return {}
        total_num = len(label_set)
        predict = self.predict(feature_set)
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
        for _sample_ in range(total_num):
            if predict[_sample_] == label_set[_sample_]:
                if label_set[_sample_] == positive:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if label_set[_sample_] == positive:
                    false_negative += 1
                else:
                    false_positive += 1
        print("tp=",true_positive,"tn=",true_negative,"fn=",false_negative,"fp=",false_positive)
        accuracy = (true_positive + true_negative) / total_num  # 准确率（预测正确的样本与总样本数之比）
        precision = true_positive / (true_positive + false_positive)  # 精度（所有 预测 为阳性的样本中， 真值 为阳性的比例）
        recall = true_positive / (true_positive + false_negative)  # 召回率（所有 真值 为阳性的样本中， 预测 为阳性的比例）
        specificity = true_negative / (true_negative + false_positive)  # 特异性（所有 真值 为阴性的样本中， 预测 为阴性的比例）
        F1_Score = (2 * precision * recall) / (precision + recall)  # 精度与召回率的加权平均
        print("accuracy:", accuracy, "precision:", precision, "recall:", recall, "specificity:",specificity, "F1_Score:", F1_Score)
        

    # 获取某一类的样本中心点
    def get_center(self, key):
        if key in self.center_dict.keys():
            return self.center_dict[key]
        else:
            return []

    def get_center_dict(self):
        return self.center_dict
#end 

#画分割线

# 展示二维平面上，二分类问题的决策线（class_1和class_2）
# feature是样本特征集合，label是对应的标签集合，对每一维特征进行两两比较，n表示特征维数
def show_decision_line(feature, label, med_classifier, class_1=0, class_2=0, n=0):
    plt.figure(figsize=(16, 12), dpi=80)  # 整张画布大小与分辨率
    img = [[] for i in range(n * n)]
    for i in range(n):
        for j in range(n):
            img[i * n + j] = plt.subplot(n, n, i * n + j + 1)
            center_1 = med_classifier.get_center(class_1)
            center_2 = med_classifier.get_center(class_2)
            c_1 = [center_1[i], center_1[j]]  # class_1类中心点的i, j两维的分量
            c_2 = [center_2[i], center_2[j]]  # class_2类中心点的i, j两维的分量
            center_3 = [(c_1[0] + c_2[0]) / 2, (c_1[1] + c_2[1]) / 2]  # 两点连线的中点
            k2, b2 = calculate_vertical_line(c_1, c_2)  # 两点中垂线的斜率和截距
            plt.scatter(feature[:, i], feature[:, j], c=label, s=20, marker='.')  # 整个样本集在特征0和2上的散点图
            plt.scatter(c_1[0], c_1[1], c='b', marker='x')  # 显示med分类器计算的样本中心点
            plt.scatter(c_2[0], c_2[1], c='b', marker='x')
            plt.grid(True)  # 显示网格线
            plt.axis('equal')  # 横纵坐标间隔大小相同
            plt.axline(c_1, c_2, color='c', linestyle="--", label="connected line")
            plt.axline(center_3, slope=k2, color='r', label="decision line")
            if i == j:
                plt.legend()  # 对角线上的子图显示出图例
            plt.xlabel("feature " + str(i))
            plt.ylabel("feature " + str(j))
            plt.tight_layout()  # 自动调整子图大小，减少相互遮挡的问题
    plt.show()


# 计算两点连线，返回斜率和纵截距（假设是二维平面上的点，并且用列表表示）
def calculate_connected_line(point_1, point_2):
    if len(point_1) != 2 or len(point_2) != 2:
        return None
    k = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
    b = (point_1[0] * point_2[1] - point_2[0] * point_1[1]) / (point_1[0] - point_2[0])
    return k, b


# 计算两点中垂线，返回斜率和纵截距（假设是二维平面上的点，并且用列表表示）
def calculate_vertical_line(point_1, point_2):
    if len(point_1) != 2 or len(point_2) != 2:
        return None
    k = -(point_1[0] - point_2[0]) / (point_1[1] - point_2[1])
    b = (point_1[1] + point_2[1] + (point_1[0] + point_2[0]) * (point_1[0] - point_2[0]) / (point_1[1] - point_2[1]))/2
    return k, b
#画分割线end

# feature表示样本特征，label表示对应的标签,m行n列共计m*n个子图
def visualization(feature, label, m, n):
    plt.figure(figsize=(10, 10), dpi=100)
    img = [[] for i in range(m*n)]
    for i in range(m):
        for j in range(n):
            img[i*n+j] = plt.subplot(m, n, i*n+j+1)
            plt.xlabel("x"+str(i))
            plt.ylabel("x"+str(j))
            plt.xlim(-1, 9)
            plt.ylim(-1, 9)     
            plt.scatter(feature[:, i], feature[:, j], s=5, c=label, marker='x')
            plt.grid(True)  # 显示网格线
            plt.tight_layout()  # 自动调整子图大小，减少相互遮挡的问题
    plt.show()

# feature表示样本特征，label表示对应的标签,m行n列共计m*n个子图
def visualization_white(feature, label, m, n):
    plt.figure(figsize=(10, 10), dpi=100)
    img = [[] for i in range(m*n)]
    for i in range(m):
        for j in range(n):
            img[i*n+j] = plt.subplot(m, n, i*n+j+1)
            plt.xlabel("x"+str(i))
            plt.ylabel("x"+str(j))
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)        
            plt.scatter(feature[:, i], feature[:, j], s=5, c=label, marker='x')
            plt.grid(True)  # 显示网格线
            plt.tight_layout()  # 自动调整子图大小，减少相互遮挡的问题
    plt.show()

# 去除某个类别的样本，返回两个numpy数组
def remove_from_data(feature, label, num):
    new_feature = []
    new_label = []
    for index in range(len(label)):
        if label[index] != num:
            new_feature.append(feature[index])
            new_label.append(label[index])
    return np.asarray(new_feature), np.asarray(new_label)


# 特征白化，返回白化后的矩阵（numpy数组格式）
# 参数为numpy格式的数组，其格式为数学上的矩阵的转置
def whitening(data):
	Ex=np.cov(data,rowvar=False) #Ex为data的协方差矩阵
	print(Ex.shape)
	a, b = np.linalg.eig(Ex) #原始特征协方差矩阵Ex的特征值和特征向量
	#特征向量单位化
	modulus=[]
	b=np.real(b)
	for i in range(b.shape[1]):
		sum=0
		for j in range(b.shape[0]):
			sum+=b[i][j]**2
		modulus.append(sum)
	modulus=np.asarray(modulus,dtype="float64")
	b=b/modulus
	#对角矩阵A
	a=np.real(a)
	A=np.diag(a**(-0.5))
	W=np.dot(A,b.transpose())
	X=np.dot(W,np.dot(Ex,W.transpose()))
	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			if np.isnan(W[i][j]):
				W[i][j]=0
	print(W)
	return np.dot(data,W)


if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_target = iris.target
    iris_target_names=iris.target_names
    print(iris)
    #可视化  
    visualization(iris_data,iris_target,4,4)  

    #去除线性不可分的最后一个
    iris_data_linear, iris_target_linear = remove_from_data(iris_data, iris_target, 2)
    visualization(iris_data_linear,iris_target_linear,4,4)  
    #划分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data_linear, iris_target_linear, test_size=0.3)
    meds=Medclass()
    meds.train(x_train,y_train)
    meds.performance(x_test, y_test, 0)
    # 展示每个特征两两对比图，显示决策线
    show_decision_line(x_test, y_test, meds, class_1=0, class_2=1, n=4)

    #特征白化
    iris_data_white = whitening(iris_data) 
    print(iris_data_white)
    visualization_white(iris_data_white,iris_target,4,4) 

    #去除线性可分的类
    #iris_data_nolinear, iris_target_nolinear = remove_from_data(iris_data, iris_target, 0) #无白化
    #visualization(iris_data_nolinear,iris_target_nolinear,4,4) 
    iris_data_nolinear, iris_target_nolinear = remove_from_data(iris_data_white, iris_target, 0)#白化
    visualization_white(iris_data_nolinear,iris_target_nolinear,4,4)  
    #划分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data_nolinear, iris_target_nolinear, test_size=0.3)
    meds2=Medclass()
    meds2.train(x_train,y_train)
    meds2.performance(x_test, y_test, 1)
    # 展示每个特征两两对比图，显示决策线
    show_decision_line(x_test, y_test, meds2, class_1=1, class_2=2, n=4)
