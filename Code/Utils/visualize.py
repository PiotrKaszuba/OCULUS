import matplotlib.pyplot as plt

import Code.Libraries.MyOculusCsvLib as mocl
import numpy as np
import itertools


def visualizePlot(scoreIndex, averaging=False):
    scores = mocl.getScoresList("../../weights/scoresSAB700.csv")
    names = []
    metr = ['Youden', 'Jaccard', 'Dice']
    data = []
    base_name = scores[0][0][9:]
    for score in scores:
        if score[0][9:16] not in names:
            data.append([float(x) for x in score[1:]])
            tempName = score[0][10:-2]
            #tempName = tempName[:5] + tempName[-1]
            names.append(tempName)
    l = (sorted(zip(names, data), key=lambda pair: pair[0]))

    names, data = [list(t) for t in zip(*l)]
    #names, data = [y for x, y in (sorted(zip(names, data), key=lambda pair: pair[0]))]
    data = [['%.3f' % j for j in i] for i in data]

    chunks = 11
    data = [data[i:i + chunks] for i in range(0, len(data), chunks)]

    names = [names[i:i + chunks] for i in range(0, len(names), chunks)]

    color = ['b', 'm', 'g', 'r', 'c', 'k']
    sums = [[0, 0, 0, 0] for x in range(chunks)]

    for i in range(len(data)):
        print(i)
        name = names[i][0]
        if name == '':
            name = base_name
        named = False
        for k in scoreIndex:
            temp = []
            for j in range(len(data[i])):
                if j == 0:
                    continue
                temp.append(float(data[i][j][k]))
                sums[j][k] = sums[j][k] + float(data[i][j][k])
            l = list(np.arange(chunks-1) * 100+100)
            if not named:
                plt.plot(l, temp, label=names[i][0], color=color[i])
                named = True
            else:
                plt.plot(l, temp, color=color[i])
    if averaging:
        for k in scoreIndex:
            temp = []
            for j in range(len(sums)):
                if j == 0:
                    continue
                sums[j][k] /= len(data)
                sums[j][k] = float('%.3f' % sums[j][k])
                temp.append(sums[j][k])
            l = list(np.arange(chunks - 1) * 100 + 100)
            plt.plot(l, temp, label='Average', color='k')
    sums = sums[1:]
    plt.legend()
    plt.title("Dystans wykrytego środka od prawidłowego")
    plt.text(x=475, y=1.094, s='Ilość epok', fontdict={'size': 10})
    #plt.title("Indeks Youdena")
    #plt.text(x=475, y=0.775, s='Ilość epok', fontdict={'size': 10})
    #plt.title("Indeks Jaccarda")
    #plt.text(x=475, y=0.594, s='Ilość epok', fontdict={'size': 10})
    #plt.title("Współczynnik Dice'a")
    #plt.text(x=475, y=0.72, s='Ilość epok', fontdict={'size': 10})
    #plt.text(x=950, y=3.460, s='Ilość epok', fontdict={'size': 10})

    fig, axs = plt.subplots(2, 1)
    axs[0].axis('tight')
    axs[0].axis('off')
    #table = axs[0].table(colLabels=['Youden', 'G. Youden', 'G. Jaccard', 'G. Dice'], loc='center', cellText=sums, rowLabels=[str(x) for x in np.arange(20) * 100 +100],
    #                      rowLoc='right', cellLoc='right', cellColours=[['w']*4]*10, rowColours=['w', 'w', 'g', 'w', 'w']*4
    #                     )
    #table.set_fontsize(12)
   # table.scale(0.8, 1.4)
    plt.show()

def visualizePlot2(scoreIndex, averaging=False):
    scores = mocl.getScoresList("../../weights/scoreswn.csv")
    names = []
    metr = ['Youden', 'Jaccard', 'Dice']
    data = []
    base_name = scores[0][0][9:16]
    for score in scores:
        if score[0][9:] not in names:
            data.append([float(x) for x in score[1:]])
            tempName = score[0][15:-5]
            #tempName = tempName[:5] + tempName[-1]
            names.append(tempName)
    l = (sorted(zip(names, data), key=lambda pair: pair[0]))

    names, data = [list(t) for t in zip(*l)]
    #names, data = [y for x, y in (sorted(zip(names, data), key=lambda pair: pair[0]))]
    data = [['%.3f' % j for j in i] for i in data]

    chunks = 21
    data = [data[i:i + chunks] for i in range(0, len(data), chunks)]

    names = [names[i:i + chunks] for i in range(0, len(names), chunks)]

    color = ['b', 'm', 'g', 'r', 'c', 'k']
    sums = [[0, 0, 0, 0] for x in range(chunks)]

    for i in range(len(data)):
        print(i)
        name = names[i][0]
        if name == '':
            name = base_name
        named = False
        for k in scoreIndex:
            temp = []
            for j in range(len(data[i])):
                if j == 0:
                    continue
                temp.append(float(data[i][j][k]))
                sums[j][k] = sums[j][k] + float(data[i][j][k])
            l = list(np.arange(chunks-1) * 100+100)
            if not named:
                plt.plot(l, temp, label=names[i][0], color=color[k])
                named = True
            else:
                plt.plot(l, temp, color=color[i])
    if averaging:
        for k in scoreIndex:
            temp = []
            for j in range(len(sums)):
                if j == 0:
                    continue
                sums[j][k] /= len(data)
                sums[j][k] = float('%.3f' % sums[j][k])
                temp.append(sums[j][k])
            l = list(np.arange(chunks - 1) * 100 + 100)
            plt.plot(l, temp, label='Average', color='k')
    sums = sums[1:]
    plt.legend()
    plt.title("Dystans wykrytego środka od prawidłowego")
    plt.text(x=950, y=3.460, s='Ilość epok', fontdict={'size': 10})

    fig, axs = plt.subplots(2, 1)
    axs[0].axis('tight')
    axs[0].axis('off')
    table = axs[0].table(colLabels=['Youden', 'G. Youden', 'G. Jaccard', 'G. Dice'], loc='center', cellText=sums, rowLabels=[str(x) for x in np.arange(20) * 100 +100],
                          rowLoc='right', cellLoc='right', cellColours=[['w']*4]*20, rowColours=['w', 'w', 'g', 'w', 'w']*4
                         )
    table.set_fontsize(12)
    table.scale(0.8, 1.4)
    plt.show()


def visualizeTable(columns=['Dystans', 'Youden', 'Jaccard', 'Dice']):
    scores = mocl.getScoresList("../../weights/scores.csv")
    names = []

    data = []

    for score in scores:
        data.append([float(x) for x in score[1:]])
        names.append(score[0][10:-4])
    data = [y for _, y in (sorted(zip(names, data), key=lambda pair: int(pair[0][-3:])))]
    data = [['%.3f' % j for j in i] for i in data]

    fig, axs = plt.subplots(2, 1)
    axs[0].axis('tight')
    axs[0].axis('off')
    table = axs[0].table(colLabels=columns, loc='center', cellText=data, rowLabels=names,
                         rowColours=['gray', 'red'] * 3, rowLoc='right', cellLoc='right',
                         cellColours=[['gray'] * 4, ['red'] * 4] * 3)
    table.set_fontsize(12)
    table.scale(0.8, 1.4)
    for i in range(1, len(columns)):
        tempDat = []
        tempDat2 = []
        for j in range(int(len(data) / 2)):
            tempDat.append(float(data[j * 2 + 1][i]))
            tempDat2.append(float(data[j * 2][i]))
        if i == 1:
            axs[1].plot([300, 500, 700], tempDat, 'r-', label='SAB')
            axs[1].plot([300, 500, 700], tempDat2, 'k-', label='Gray')
        else:
            axs[1].plot([300, 500, 700], tempDat, 'r-')
            axs[1].plot([300, 500, 700], tempDat2, 'k-')
    plt.title("Youden, Dice, Jaccard (kolejno od góry)")
    plt.legend()
    plot_margin = 0.1

    x0, x1, y0, y1 = plt.axis()
    axs[1].axis((x0 - plot_margin,
                 x1 + plot_margin,
                 y0,
                 y1))
    plt.text(x=400, y=0.58, s='Liczebność zbioru treningowego', fontdict={'size': 10})
    plt.show()


def visualizeBar(scoreIndex, title):
    scores = mocl.getScoresList("../../weights/scores.csv")
    vals = []
    names = []
    for score in scores:
        vals.append(float(score[scoreIndex]))
        names.append(score[0][10:-4])

    l = (sorted(zip(names, vals), key=lambda pair: int(pair[0][-3:])))

    names, vals = [list(t) for t in zip(*l)]

    plt.bar(names, vals, color=['gray', 'red'])

    plt.title(title)
    # plt.text(x=-1.5, y=-1.3, s='500 epok, redukcja kroku uczenia', fontdict={'size':8})
    # plt.text(x=4, y=13.3, s='Gray vs SAB + zbiór treningowy', fontdict={'size':8})
    # plt.xticks(y_pos, names)
    plt.show()


if __name__ == "__main__":
    #visualizeBar(1, "Dystans wykrytego środka od prawidłowego")
    #visualizeTable()
    visualizePlot2([0], False)
    #visualizeTable()