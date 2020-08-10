import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('C:\\vscode\py\python_pigeon_farm\py_panda\电影信息.xlsx')

#print(df)

#接下来我们将对数据进行表示：将所有数据用由元组组成的列表表示
pairs=[]
for i in range(len(df)):
    actors=df.at[i,'演员'].split('，')
    for actor in actors:
        pair=(actor,df.at[i,'电影名称'])
        pairs.append(pair)

pairs=sorted(pairs,key=lambda item:int(item[0][2:]))
#print(pairs)

#下面我们整理新的表格：
index=[item[0] for item in pairs]
data=[item[1] for item in pairs]
df1=pd.DataFrame({'演员':index,'电影名称':data})

result = df1.groupby('演员', as_index=False, sort=False).count()
result.columns=('演员','电影数')
result.to_excel('C:\\vscode\py\python_pigeon_farm\py_panda/演员统计表.xlsx')
#print(result)

plt.figure(figsize=(5,5),dpi=100,facecolor='LightPink')
result.plot(x='演员',y='电影数',kind='bar')
plt.savefig('C:\\vscode\py\python_pigeon_farm\py_panda/演员统计表.png')
plt.show()
