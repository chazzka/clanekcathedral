from ai.trainer import loadModel, predict, getClusterLabels, doTrain, localOutlierTrain

from mock.randomdatagenerator import *

from articletester.articletester import *

import sys
import tomli


def getConfigFile(path):
    with open(path, mode="rb") as fp:
        return tomli.load(fp)


if __name__ == "__main__":


    # train
    trainData = list(createRandomData())

    trainData = list(generateRandomData([
        #generateRandomClusters(centers=[(80, 200), (100,200), (20,200)]),
        generateLinearSpace(leftBoundary=50,rightBoundary=100),
        generateLinearSpace(leftBoundary=0,rightBoundary=40),
        #generateLinearSpace(20, s),
        #generateLinearSpace(20, r),
    ]))

    testData = list(generateRandomData([
        generateRandomClusters(centers=[(80, 200), (100,200), (20,200)]),
        generateLinearSpace(leftBoundary=60,rightBoundary=110),
        generateLinearSpace(leftBoundary=0,rightBoundary=50),
        #generateLinearSpace(20, s),
        #generateLinearSpace(20, r),
    ]))

    trained_model = localOutlierTrain(trainData)

    predictedList = predict(
        testData,
        trained_model
    )

    #clusters = getClusterLabels(xyValues, predictedList, config["AI"])

    fi, ax = plotXYData(*zip(*testData), plt.subplots(), c=list(map(lambda x: 'white' if x == -1 else 'black',predictedList)), savepath="figures/lofres.svg")

    print("done")
    sys.exit(0)