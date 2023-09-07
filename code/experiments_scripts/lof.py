from mock.randomdatagenerator import *
import sys
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
import collections
import functools
import operator

from sklearn.neighbors import LocalOutlierFactor

from articletester.articletester import *


def train_predict(trainData, testData):
    clf = LocalOutlierFactor(novelty=True).fit(trainData)
    predictedList = clf.predict(testData)

    regular = testData[predictedList == 1]

    outlier = testData[predictedList == -1]

    return {"regular": len(regular), "outlier": len(outlier)}


def plot_one_run(trainData, testData):
    clf = LocalOutlierFactor(novelty=True).fit(trainData)
    predictedList = clf.predict(testData)

    regular = testData[predictedList == 1]

    outlier = testData[predictedList == -1]

    fi, ax = plotXYDataTwoClasses(regular, outlier, plt.subplots(), c=['black', 'white'], labels=[
                                  "Regular", "Outlier"], savepath="figures/lof_experiment6.svg")


if __name__ == "__main__":

    trainData = np.loadtxt('experiments_data/train.out', delimiter=',')
    testData = np.loadtxt('experiments_data/experiment6.out', delimiter=',')

    n_runs = 30

    regular_outlier_counts = list(
        map(lambda x: train_predict(trainData, testData), range(n_runs)))

    regular_outlier_average = np.asarray(list(dict(functools.reduce(
        operator.add, map(collections.Counter, regular_outlier_counts))).values()))

    print(regular_outlier_average/n_runs)

    plot_one_run(trainData, testData)

    print("done")
    sys.exit(0)
