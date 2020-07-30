#python画图
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

x=np.arange(0,5,0.02)

plt.figure(figsize=(5,8),dpi=120,facecolor='LightPink',edgecolor='LightPink')

plt.subplot(2,1,1)
plt.plot(x,f(x))
plt.xlabel('自变量')
plt.ylabel('因变量')
plt.title('测试函数')

plt.subplot(2,1,2)
plt.plot(x,x*x,'r--')
plt.xlabel('自变量')
plt.ylabel('因变量')

plt.savefig('C:\\vscode\py\python_pigeon_farm\py_matplotlib/test.png')
plt.show()
