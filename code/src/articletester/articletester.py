import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from itertools import *
from toolz import groupby, unique


# @deprecated, use plotXYWithMean
def plotDataWithMean(xyData):
    # todo: toto lepsi
    yValues = list(map(lambda x: x[1] ,xyData))
    mean = np.mean(yValues)
    grouped = groupby(lambda x: x[1] > mean, xyData)

    print(grouped)
    
    fig = plt.figure()
    ax1 = fig.add_subplot()

    ax1.scatter(*zip(*grouped[True]), c='green')
    ax1.scatter(*zip(*grouped[False]), c='blue')
    ax1.axhline(mean, c='red')
    plt.legend(loc='upper left')
    plt.xlabel("Time")
    plt.ylabel("Observed value")
    plt.show()


def plotXYData(regular, fig_ax_tuple, savepath = 0, c='red', labels=["Regular"]):
    print(fig_ax_tuple)
    fig = fig_ax_tuple[0]
    ax = fig_ax_tuple[1]

    scatter1 = ax.scatter(*zip(*regular), c=c[0],  facecolors='w', edgecolors='black', label=labels[0])
    ax.legend()
    
    plt.xlabel("Time", loc='right')
    plt.ylabel("Observed value", loc='top')
    
    if savepath:
        plt.savefig(savepath, format="svg")

    return fig, ax


def plotXYDataTwoClasses(regular, outlier, fig_ax_tuple, savepath = 0, c='red', labels=["Regular", "Outlier"]):
    print(fig_ax_tuple)
    fig = fig_ax_tuple[0]
    ax = fig_ax_tuple[1]

    if len(regular):
        scatter1 = ax.scatter(*zip(*regular), c=c[0],  facecolors='w', edgecolors='black', label=labels[0])

    if len(outlier):    
        scatter2 = ax.scatter(*zip(*outlier), c=c[1],  facecolors='w', edgecolors='black', label=labels[1])
    ax.legend()
    
    plt.xlabel("Time", loc='right')
    plt.ylabel("Observed value", loc='top')
    
    if savepath:
        plt.savefig(savepath, format="svg")

    return fig, ax


def plotYData(yvalue, fig_ax_tuple):
    
    fig = fig_ax_tuple[0]
    ax = fig_ax_tuple[1]
    ax.axhline(yvalue)

    return fig, ax


def plotXYWithMean(trainXyValues, savepath = 0):
    mean = np.mean(list(map(lambda x: x[1] ,trainXyValues)))

    group = dict(groupby(lambda x: x[1] > mean, trainXyValues))

    fi, ax = plotXYDataTwoClasses(group[False], group[True], plt.subplots(), c=['black', 'white'], labels=[
                                  "Regular", "Outlier"], savepath=savepath)    

    finalfig, finalax = plotYData(mean, (fi, ax))
    if savepath:
        plt.savefig(savepath, format="svg")
    else:
        plt.show()
