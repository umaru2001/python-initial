import numpy as np
import operator

def createDataset():
    group=np.array([[1.0,0.9],[1.2,1.0],[0.1,0.2],[0.1,0.3]])
    labels=['A','A','B','B']
    return group,labels

def kNNClassify(newInput,dataSet,labels,k):
    numSamples=dataSet.shape[0]
    diff=np.tile(newInput,(numSamples,1))-dataSet
    distance=np.sum(diff**2,axis=1)**0.5
    sortedDistIndices=np.argsort(distance)

    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1

    maxIndex=0
    for key,value in classCount.items():
        if value>maxIndex:
            maxIndex=value
            maxKey=key
    return maxKey

dataSet,labels=createDataset()
newInput=(1.0,1.0)
k=3
print("您的输入是：",newInput)
print("该数据分类为：",kNNClassify(newInput,dataSet,labels,k))

