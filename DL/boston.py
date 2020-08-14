import numpy as np
import json
import matplotlib.pyplot as plt

def load_data():
    datafile = 'C:\\vscode\py\python_pigeon_farm\DL\housing.data'
    data = np.fromfile(datafile, sep=' ')
    #print(data)

    feature_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', \
                    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num=len(feature_names)
    data=data.reshape((data.shape[0]//feature_num,feature_num))
    #print(data[0])
    #print(data.shape)

    ratio=0.8
    #offset是训练集的数量
    offset=int(data.shape[0]*ratio)
    train_data=data[:offset]
    #接下来进行数据归一化处理：
    maximums,minimums,avgs=\
        train_data.max(axis=0),\
        train_data.min(axis=0),\
        train_data.sum(axis=0)/train_data.shape[0]
    #对数据进行归一化处理：
    data=(data-avgs)/(maximums-minimums)
    #检查归一化是否成功
    #print(train_data[0])
    train_data=data[:offset]
    test_data=data[:offset]
    return train_data,test_data

class Network(object):
    def __init__(self,num_of_weights):
        #随机产生随机数w，设定同一个种子
        #w表示的是随机产生的参数
        np.random.seed(0)
        self.w=np.random.randn(num_of_weights,1)
        self.b=0

    def forward(self,x):
        z=np.dot(x,self.w)+self.b
        return z
    
    def loss(self,z,y):
        cost=(z-y)**2
        return np.sum(cost)/cost.shape[0]
    
    def gradient(self,x,y):
        z=self.forward(x)
        gradient_w=(z-y)*x
        gradient_w=np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:,np.newaxis]
        gradient_b=(z-y)
        gradient_b=np.mean(gradient_b)
        return gradient_w,gradient_b

    def update(self,gradient_w,gradient_b,eta=0.01):
        self.w-=eta*gradient_w
        self.b-=eta*gradient_b
    
    def train(self,x,y,iterations=100,eta=0.01):
        losses=[]
        for i in range(iterations):
            losses.append(self.loss(self.forward(x),y))
            gradient_w,gradient_b=self.gradient(x,y)
            self.update(gradient_w,gradient_b,eta)
        return losses,self.w,self.b    


train_data,test_data=load_data()
x=train_data[:,:-1]
y=train_data[:,-1:]
net=Network(x.shape[1])
#forward=net.forward(x)
#loss=net.loss(forward(x),y)
#下面是计算显示损失函数：
#print(net.loss(z,y)[0])
#下面是同时计算404个数据使用：
#print(net.forward(x))
#gradient_w,gradient_b=net.gradient(x,y)
#print(gradient_w[0],gradient_b[0])
num_iterations=2000
eta=0.01
losses,w,b=net.train(x,y,iterations=num_iterations,eta=0.01)

plot_x=np.arange(num_iterations)
plot_y=np.array(losses)
plt.plot(plot_x,plot_y)
plt.savefig('C:\\vscode\py\python_pigeon_farm\DL\loss.png')
plt.show()
print(w,b)