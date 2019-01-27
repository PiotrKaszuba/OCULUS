import matplotlib.pyplot as plt

import Code.Libraries.MyOculusCsvLib as mocl
import numpy as np
scores = mocl.getScoresList("../../weights/scores.csv")


def visualizePlot(scoreIndex):
    names = []

    data = []
    for score in scores:
        if score[0][10:] not in names:
            data.append([float(x) for x in score[1:]])
            names.append(score[0][10:])

    data = [y for _, y in (sorted(zip(names, data), key=lambda pair: int(pair[0][-3:])))]
    data = [['%.3f' % j for j in i] for i in data]

    chunks = 3
    data = [data[i:i + chunks] for i in range(0, len(data), chunks)]

    names = [names[i:i + chunks] for i in range(0, len(names), chunks)]

    for i in range(len(data)):
        temp = []
        name = names[i][0]
        for j in range(len(data[i])):
            temp.append(float(data[i][j][scoreIndex]))
        l = list(np.arange(chunks) * 200 + 300)
        plt.plot(l, temp)
    plt.show()




def visualizeTable(columns=['Dystans', 'Youden', 'Jaccard', 'Dice']):
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
    # visualizeBar(1, "Średni dystans od środka tarczy")
    # visualizeTable()
    visualizePlot(0)
