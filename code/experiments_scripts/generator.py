from mock.randomdatagenerator import *
import sys
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from articletester.articletester import *

if __name__ == "__main__":


    testData = list(generateRandomData([
        #generateRandomClusters(centers=[(80, 200)], n_samples=10),
        generateLinearSpace(leftBoundary=60,rightBoundary=110),
        generateLinearSpace(leftBoundary=0,rightBoundary=50),
        generateRandomClusters(centers=[(55, 200)], n_samples=10)
        #generateLinearSpace(20, s),
        #generateLinearSpace(20, r),
    ]))

    testData = np.asarray(testData)


    np.savetxt('experiments_data/train.out', testData, delimiter=',')

    fi, ax = plotXYData(testData, plt.subplots(), c=[
                                  'black'], labels=[""], savepath=f"figures/data_overview.svg")

    plt.show()

    print("done")
    sys.exit(0)
