from matplotlib.pyplot import plot
from numpy.core.multiarray import result_type
import paddle
from paddle import batch
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from paddle.fluid.framework import set_flags

trainset = paddle.dataset.mnist.train()
train_reader = paddle.batch(trainset, batch_size=8)
'''
# 以迭代的形式读取数据
for batch_id, data in enumerate(train_reader()):
    # 获得图像数据，并转为float32类型的数组
    img_data = np.array([x[0] for x in data]).astype('float32')
    # 获得图像标签数据，并转为float32类型的数组
    label_data = np.array([x[1] for x in data]).astype('float32')
    # 打印数据形状
    #print("图像数据形状和对应数据为:", img_data.shape, img_data[0])
    #print("图像标签形状和对应数据为:", label_data.shape, label_data[0])
    break

print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(label_data[0]))
# 显示第一batch的第一个图像
img = np.array(img_data[0]+1)*127.5
img = np.reshape(img, [28, 28]).astype(np.uint8)

plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()
'''

class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()
        self.fc=Linear(input_dim=784,output_dim=1,act=None)

    def forward(self,inputs):
        outputs=self.fc(inputs)
        return outputs
'''
with fluid.dygraph.guard():
    model=MNIST()
    model.train()
    train_loader=paddle.batch(paddle.dataset.\
        mnist.train(),batch_size=16)
    optimizer=fluid.optimizer.SGDOptimizer\
        (learning_rate=0.001,parameter_list=model.parameters(),)
    
    EPOCH_MUN=10
    losses_list=[]
    for epoch_id in range(EPOCH_MUN):
        for batch_id,data in enumerate(train_loader()):
            image_data=np.array([x[0] for x in data]).astype('float32')
            label_data=np.array([x[1] for x in data]).astype('float32').reshape(-1,1)

            image=fluid.dygraph.to_variable(image_data)
            label=fluid.dygraph.to_variable(label_data)

            predict=model(image)

            loss=fluid.layers.square_error_cost(predict,label)
            avg_loss=fluid.layers.mean(loss)

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

            if batch_id !=0 and batch_id  % 1000 == 0:
                print("epoch: {0}, batch: {1}, loss is: {2}".format(epoch_id, batch_id, avg_loss.numpy()))

fluid.save_dygraph(model.state_dict(), 'mnist_1.0')
'''
def load_image():
    img=Image.open('C:\\vscode\py\python_pigeon_farm\picture_recognition\example_0.png').convert('L')
    img=img.resize((28,28),Image.ANTIALIAS)
    img_st=np.array(img).reshape(1,-1).astype(np.float32)
    img_st=1-img_st/127.5
    return img_st

with fluid.dygraph.guard():
    model=MNIST()
    params_file_path='C:\\vscode\py\python_pigeon_farm\picture_recognition\mnist_1.0.pdparams'
    
    model_dict, _ = fluid.load_dygraph(params_file_path)
    model.load_dict(model_dict)

    model.eval()
    tensor_img=load_image()
    result=model(fluid.dygraph.to_variable(tensor_img))

    print("本次预测的数字是", result.numpy().astype('int32'))