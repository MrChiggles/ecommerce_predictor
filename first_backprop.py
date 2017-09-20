import matplotlib.pyplot as plt
import numpy as np

def forwardProp(X, W1, b1, W2, b2):
    # we use tanh here as it achieves a very similar effect to sigmoid
    # and is done in one line
    Z = np.tanh(X.dot(W1) + b1)

    A = Z.dot(W2) + b2
    expA = np.exp(A)

    # The softmax function spreads each y value across a 1 probability
    # We then take the highest one as the 'selected' value by the computer
    Y = expA / expA.sum(axis=1, keepdims=True)

    return Y, Z

def derivative_w2(hidden, predicted, output):

    # N, K = predicted.shape
    # M = hidden.shape[1]

    # N => 1500
    # K => 3
    # M => 3

    # ret1 = np.zeros((M, K))

    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             ret1[m, k] += (predicted[n, k] - output[n, k]) * hidden[n, m]

    # we ultimately just want to get the partial derivative of how far our prediction was off from the true output
    # IE: the delta error
    # this should have dimensions: 3 x 3

    # The output layer has 1 x 3 dimensions / 1500 x 3
    # the hidden layer has 1 x 3 dimensions / 1500 x 3
    # 1500 because we have 1500 in our sample set

    # transposing the hidden matrix turns it into a
    # 3 x 1500 which is then multipliable by the difference our predicted & output
    # giving us 3 x 3 which is the size required, leaving only the values to be tested

    # this can be verified with an assert and uncommenting the above for loop
    return hidden.T.dot(predicted - output)

def derivative_b2(predicted, output):
    # this will return the 'error' of the bias
    return (predicted - output).sum(axis=0)

def derivative_w1(X, hidden, predicted, output, W2):

    # our input layer should have 1500 x 2 dimensions
    # N, D = X.shape
    # our W2 layer should have 2 x 3 dimensions
    # M, K = W2.shape

    # 2 x 3 shape
    # ret1 = np.zeros((D, M))

    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             for d in range(D):
    #                 ret1[d, m] += (predicted[n, k] - output[n, k]) * W2[m, k] * hidden[n, m] * (1-hidden[n, m]) *
    # X[n, d]

    return X.T.dot((predicted - output).dot(W2.T) * (hidden * (1 - hidden)))

def derivative_b1(predicted, output, W2, hidden):
    return ((predicted - output).dot(W2.T) * hidden * (1 - hidden)).sum(axis=0)

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0

    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1

    return float(n_correct / n_total)

def cost(indicator, output):
    total = indicator * np.log(output)
    return total.sum()

def main():

    Nclass = 500
    D = 2
    M = 3
    K = 3

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
    N = len(Y)

    indicator = np.zeros((N, K))
    for i in range(N):
        indicator[i, Y[i]] = 1

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)

    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    alpha_learning_rate = 10e-7
    costs_record = []

    for epoch in range(100000):
        output, hidden = forwardProp(X, W1, b1, W2, b2)

        if epoch % 100 == 0:
            c = cost(indicator, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print('Cost: ', c, ' classification rate: ', r)
            costs_record.append(c)

        W2 += alpha_learning_rate * derivative_w2(hidden, indicator, output)
        b2 += alpha_learning_rate * derivative_b2(indicator, output)
        W1 += alpha_learning_rate * derivative_w1(X, hidden, indicator, output, W2)
        b1 += alpha_learning_rate * derivative_b1(indicator, output, W2, hidden)

    plt.plot(costs_record)
    plt.show()

if __name__ == '__main__':
    main()
