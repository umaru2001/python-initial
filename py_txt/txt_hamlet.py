#python文本处理
#统计哈姆雷特中出现最多的十个词：
f=open('C:/vscode\py\py_txt\hamlet.txt','r')
#以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。
txt=f.read()
txt=txt.lower();
for c in '\'\"\\-!?$%()<>=_|{}~,;./*&@^':
    txt=txt.replace(c," ")

wordList=txt.split()
#用空格分离并以列表形式返回
wordDict={}
for word in wordList:
    wordDict[word]=wordDict.get(word,0)+1

items = list(wordDict.items())
items.sort(key=lambda x:x[1], reverse=True)

for i in range(10):
    word, count = items[i]
    print("{0:<20}({1:>5})".format(word, count))
#输出方式：<左对齐>右对齐，输出宽度

g=open('C:/vscode\py\py_txt\hamlet_derive.txt','w+')
for i in range(len(items)):
    word , count=items[i]
    g.write("{0:<20}{1:>5}\n".format(word,count))
g.close()