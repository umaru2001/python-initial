#python文本处理
#统计三国演义中出现最多的十个词：
import jieba as jb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud

f=open('C:/vscode\py\py_txt/threekingdoms.txt','r',encoding='utf-8')
txt=f.read()
for c in '‘’”“\'\"\\-！？（）《》=——|{}~·，；：。/*&@^【】、':
    txt=txt.replace(c," ")

jb.add_word('玄德')
jb.add_word('孔明')
wordList=jb.cut(txt)
wordDict={}
for word in wordList:
    if(len(word)>1):
        wordDict[word]=wordDict.get(word,0)+1

items=list(wordDict.items())
items.sort(key=lambda x:x[1],reverse=True)

'''
for i in range(20):
    word,count=items[i]
    print("{0:<20}{1:>5}".format(word,count))
'''
'''
g=open('C:/vscode\py\py_txt/threekingdoms_derive_2.txt','w+',encoding='utf-8')
for i in range(len(items)):
    word,count=items[i]
    g.write("{0:<20}{1:>5}\n".format(word,count))
g.close()
'''

#接下来动手开始做词云啦
#首先要得到一个完全纯字符串的文本：
wordlist=list()
for i in range(1000):
    word,count=items[i]
    wordlist.append(word)
txt_cloud=' '.join(wordlist)

cloud_mask=np.array(Image.open("C:/vscode\py\py_txt/misaku_mikoto.png"))

wc=WordCloud(
    background_color = "white", #背景颜色
    mask = cloud_mask,          #背景图cloud_mask
    max_words=1000, 
    font_path = 'msyh.ttc',     #最大词语数目  
    #height=658,                #设置高度
    #width=573,                 #设置宽度
    max_font_size=8,         #最大字体号
    random_state=1000,          #设置随机生成状态，即有多少种配色方案    
)

myCloud=wc.generate(txt_cloud)
plt.imshow(myCloud)
plt.axis("off")
plt.show()
wc.to_file('C:/vscode\py\py_txt/misaku_1.jpg')
