import paddle
from paddle import batch
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
from PIL import Image

def load_statistics():
    data_list=[]
    testset=paddle.dataset.mnist.test()
    test_loader=paddle.batch(testset,batch_size=1)
    for data in test_loader():
        data_list.append(data)
    data_array_new=np.array(data_list)
    np.random.shuffle(data_array_new)
    data_array_new=data_array_new[:100]
    print(data_array_new.shape)
    img_data=np.array([x[0][0] for x in data_array_new])
    label_data=np.array([x[0][1] for x in data_array_new])
    return img_data,label_data

class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()
        self.fc=Linear(input_dim=784,output_dim=1,act=None)

    def forward(self,inputs):
        outputs=self.fc(inputs)
        return outputs

count=0
with fluid.dygraph.guard():
    model=MNIST()
    params_file_path='C:\\vscode\py\python_pigeon_farm\picture_recognition\mnist_1.0.pdparams'
    model_dict, _ = fluid.load_dygraph(params_file_path)
    model.load_dict(model_dict)
    img_data,label_data=load_statistics()
    model.eval()

    for i in range(100):
        result=model(fluid.dygraph.to_variable(img_data[i]))
        print("本次预测的数字是",int(result),",本次真实的数字是",label_data[i])
        if int(result)==label_data[i]:
            count+=1
    print("本次实验成功率是：",count,"%")