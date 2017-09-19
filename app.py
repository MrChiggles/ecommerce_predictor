import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

Nclass = 500

#       500x2                       2x1
x1 = np.random.randn(Nclass, 2) + np.array([0, 0])
x2 = np.random.randn(Nclass, 2) + np.array([2, 2])
x3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([x1, x2, x3])

Y = np.array([3]*Nclass + [1]*Nclass + [2]*Nclass)

plot.scatter(X[:, 0], X[:, 1], c=Y, s=10, alpha=0.5)

# plot.show()

input_layer = 2
hidden_layer = 3
output_layer = 3

W1 = np.ones([input_layer, hidden_layer])
W2 = np.ones([hidden_layer, output_layer])

b1 = np.random.randn(hidden_layer)
b2 = np.random.randn(output_layer)

def forwardProp(X, W1, b1, W2, b2):
    z1 = (1 / (1 + np.exp(-X.dot(W1) - b1)))
    A = z1.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

def classificationRate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total+=1
        if Y[i] == P[i]:
            n_correct+=1
    return float(n_correct / n_total)

P_Y_given_X = forwardProp(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

assert (len(Y) == len(P))

print("Classification Rate for randomly chosen weights:", classificationRate(Y, P))
print(df.head())