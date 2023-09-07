from ai.trainer import trainAndPredict
from mock.randomdatagenerator import *
import sys
import tomli
from itertools import product

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest as forest
from sklearn.cluster import DBSCAN as dbscan

from articletester.articletester import *

def getConfigFile(path):
    with open(path, mode="rb") as fp:
        return tomli.load(fp)


if __name__ == "__main__":
    try:
        configFile = sys.argv[1]
    except IndexError:
        configFile = "config.toml"

    config = getConfigFile(configFile)

    # training data
    trainXyValues = list(generateRandomData([
        generateRandomClusters(centers=[(25, 200),], n_samples=10),
        generateLinearSpace(leftBoundary=60,rightBoundary=110),
        generateLinearSpace(leftBoundary=0,rightBoundary=50),
        #generateLinearSpace(20, s),
        #generateLinearSpace(20, r),
    ]))


    plotXYWithMean(trainXyValues, "figures/mean2.svg")
        
    
    plt.show()


    print("done")
    sys.exit(0)
