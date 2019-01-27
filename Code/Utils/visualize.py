import matplotlib.pyplot as plt
import Code.Libraries.MyOculusCsvLib as mocl
import numpy as np

scores = mocl.getScoresList("../../weights/scores.csv")

names = []
vals = []
data =[]
for score in scores:
    data.append([float(x) for x in score[1:]])
    names.append(score[0][10:-4])
    vals.append(float(score[1]))

l = (sorted(zip(names,vals), key=lambda pair: int(pair[0][-3:])))
data = [y for _,y in (sorted(zip(names,data), key=lambda pair: int(pair[0][-3:])))]
names, vals  =  [list(t) for t in zip(*l)]
print(vals)
#y_pos = np.arange(len(names))
data =  [['%.3f' % j for j in i] for i in data]
#data = np.random.random((10,4))
columns = [ 'Dystans' , 'Youden', 'Jaccard', 'Dice']
fig, axs =plt.subplots(2,1)
axs[0].axis('tight')
axs[0].axis('off')
table = axs[0].table(colLabels=columns, loc='center', cellText=data, rowLabels = names, rowColours=['gray','red']*3, rowLoc='right', cellLoc='right', cellColours=[['gray']*4, ['red']*4]*3)
table.set_fontsize(12)
table.scale(0.8, 1.4)
#plt.bar(names, vals, color=['gray','red'])
for i in range(1,4):
    tempDat = []
    tempDat2 = []
    for j in range(int(len(data)/2)):
        tempDat.append(float(data[j*2+1][i]))
        tempDat2.append(float(data[j*2][i]))
    axs[1].plot(np.arange(2,5),tempDat)
    axs[1].plot(np.arange(2,5), tempDat2)
#plt.title("Średni dystans od środka tarczy")
#plt.text(x=-1.5, y=-1.3, s='500 epok, redukcja kroku uczenia', fontdict={'size':8})
#plt.text(x=4, y=13.3, s='Gray vs SAB + zbiór treningowy', fontdict={'size':8})
#plt.xticks(y_pos, names)
plt.show()