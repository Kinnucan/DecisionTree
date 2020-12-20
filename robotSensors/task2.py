
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, ShuffleSplit
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
import random


def readCSV(file):
    """Given a filename that is a CSV file, reads in the data."""
    fil = open(file, 'r')
    dataset = np.loadtxt(fil, delimiter=",")
    print(dataset.shape)
    return dataset


if __name__ == "__main__":
    sensorDataset = readCSV("sensorReadings24.csv")
    sensorData, sensorTarget = sensorDataset[:,:-1], sensorDataset[:,-1]
    sensorTrainData, sensorTestData, sensorTrainTarget, sensorTestTarget = train_test_split(sensorData, sensorTarget, test_size=.1)

    # Decision trees
    dTree = tree.DecisionTreeClassifier(splitter="random", )
    dTree.fit(sensorTrainData, sensorTrainTarget)
    print("Example decision tree prediction:", dTree.predict(sensorTestData[0:1]))
    print("Example decision tree target", sensorTestTarget[0])

    # MLP (Multilayer Perceptrons)
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5,5,5,5,5, 2), random_state=1, max_iter=200, activation = "tanh")
    mlp.fit(sensorTrainData, sensorTrainTarget)
    print("Example neural network prediction:", mlp.predict(sensorTestData[0:1]))
    print("Example neural network target", sensorTestTarget[0])

    dTreeTrainPredictions, dTreeTestPredictions = dTree.predict(sensorTrainData), dTree.predict(sensorTestData)
    mlpTrainPredictions, mlpTestPredictions = mlp.predict(sensorTrainData), mlp.predict(sensorTestData)



    print("Classification reports for training data:")
    print(classification_report(sensorTrainTarget, dTreeTrainPredictions))
    # print(classification_report(sensorTrainTarget, mlpTrainPredictions))

    print("Classification reports for testing data:")
    print(classification_report(sensorTestTarget, dTreeTestPredictions))
    # print(classification_report(sensorTestTarget, mlpTestPredictions))

    print("Confusion matrices for training data:")
    print(confusion_matrix(sensorTrainTarget, dTreeTrainPredictions))
    # print(confusion_matrix(sensorTrainTarget, mlpTrainPredictions))

    print("Confusion matrices for testing data:")
    print(confusion_matrix(sensorTestTarget, dTreeTestPredictions))
    # print(confusion_matrix(sensorTestTarget, mlpTestPredictions))


    print("Cross validation:")
    random.seed()
    cv = KFold(n_splits=10,shuffle=True)
    cvTree = cross_val_score(dTree, sensorData, sensorTarget, cv=cv)
    cvMLP = cross_val_score(mlp, sensorData, sensorTarget, cv=cv)
    print(cvTree, np.mean(cvTree))
    print(cvMLP, np.mean(cvMLP))
