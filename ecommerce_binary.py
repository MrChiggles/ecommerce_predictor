import numpy as np
import matplotlib.pyplot as plt
from preprocess import get_binary_data

def sigmoid(args):
    return 1 / (1 + np.exp(args))

def forwardProp(X, W, b1, V, b2):
    Z = np.tanh(X.dot(W) + b1)
    return sigmoid(Z.dot(V) + b2), Z

def cost(predicted, labels):
    return -np.mean(predicted * np.log(labels))

def classificationRate(predicted, labels):

    total = len(labels)
    count = 0

    for i in range(total):
        if predicted[i] == labels[i]:
            count+=1

    return float(count / total)

def derivative_V(predicted, labels, hidden):
    return (predicted - labels).dot(hidden)

def derivative_b2(predicted, labels):
    return predicted - labels

def derivative_W(predicted, labels, hidden, W, X):


def main():
    _X, _Y = get_binary_data()

    X = _X[:-100]
    Y = _Y[:-100]

    testX = _X[-100:]
    testY = _Y[-100:]

    Y = Y.astype(np.int32)
    testY = testY.astype(np.int32)
    alpha = 0.01

    N, D = X.shape
    K = 1
    hNeurons = 5

    W = np.random.randn((D, hNeurons))
    b1 = np.random.randn(hNeurons)
    V = np.random.randn((hNeurons, K))
    b2 = np.random.randn(K)

    epochs = 10000

    TrainCosts = []
    TestCosts = []

    for i in range(epochs):
        predicted, hidden = forwardProp(X, W, b1, V, b2)

        if i % 1000 == 0:
            testPredicted, testHidden = forwardProp(testX, W, b1, V, b2)

            trainCost = cost(predicted, Y)
            testCost = cost(testPredicted, testY)

            trainRate = classificationRate(predicted, Y)
            testRate = classificationRate(testPredicted, testY)

            TrainCosts.append(trainCost)
            TestCosts.append(testCost)

            print("Train Cost: ", trainCost, " Rate: ", trainRate)
            print("Test Cost: ", testCost, " Rate: ", testRate)

        V -= alpha * derivative_V(predicted, Y, hidden)
        b2 -= alpha * derivative_b2(predicted, Y)
        W -= alpha * derivative_W(predicted, Y, hidden, W, X)
        b1 -= alpha * derivative_b1(predicted, Y, hidden, W, X)


# The main method
if __name__ == "__main__":
    main()