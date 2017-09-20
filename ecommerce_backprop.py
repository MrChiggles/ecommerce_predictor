import matplotlib.pyplot as plt
import numpy as np
from preprocess import get_data

def softmax(args):
    eArgs = np.exp(args)
    return eArgs / eArgs.sum(axis=1, keepdims=True)

def forwardProp(input, W1, b1, W2, b2):
    Z = np.tanh(input.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

def cost(labels, predicted):
    # return (labels * np.log(predicted)).sum()
    return -np.mean(labels*np.log(predicted))

def classification_rate(labels, predicted):

    l = np.argmax(labels, axis=1)
    p = np.argmax(predicted, axis=1)

    total = len(l)
    correct = 0

    for i in range(total):
        if (l[i] == p[i]):
            correct += 1

    return float(correct / total)

def derivative_w2(hidden, labels, predicted):
    return hidden.T.dot(predicted - labels)

def derivative_b2(labels, predicted):
    # why sum on axis 0?
    return (predicted - labels).sum(axis=0)

def derivative_w1(input, hidden, labels, predicted, W2):
    # our input layer should have 1500 x 2 dimensions
    # N, D = input.shape
    # our W2 layer should have 2 x 3 dimensions
    # M, K = W2.shape

    # 2 x 3 shape
    # ret1 = np.zeros((D, M))

    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             for d in range(D):
    #                 ret1[d, m] += (predicted[n, k] - labels[n, k]) * W2[m, k] * hidden[n, m] * (1-hidden[n, m]) * input[n, d]

    # (1 - hidden * hidden)

    return input.T.dot((predicted - labels).dot(W2.T) * (1 - hidden * hidden))

def derivative_b1(labels, predicted, W2, hidden):
    return ((predicted - labels).dot(W2.T) * hidden * (1 - hidden)).sum(axis=0)

def main():

    X, Y = get_data()
    N, D = X.shape
    M = 5
    K = 4
    Y = Y.astype(np.int32)

    labels = np.zeros((N, K))

    for i in range(len(Y)):
        labels[i, Y[i]] = 1

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    alpha = 0.001
    costs = []
    epochs = 10000

    for i in range(epochs):

        predicted, hidden = forwardProp(X, W1, b1, W2, b2)

        if i % 100 == 0:
            _cost = cost(labels, predicted)
            r = classification_rate(labels, predicted)
            print('Cost: ', _cost, ' rate: ', r)
            costs.append(_cost)

        W2 -= alpha * derivative_w2(hidden, labels, predicted)
        b2 -= alpha * derivative_b2(labels, predicted)
        W1 -= alpha * derivative_w1(X, hidden, labels, predicted, W2)
        b1 -= alpha * derivative_b1(labels, predicted, W2, hidden)

    plt.plot(costs)
    plt.show()

if __name__ == "__main__":
    main()