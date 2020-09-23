import numpy as np
from scipy import io

data = io.loadmat('spamData.mat')

xtrain = np.array(data['Xtrain'])
xtest = np.array(data['Xtest'])
ytrain = np.array(data['ytrain'])
ytest = np.array(data['ytest'])


xtrain_log = np.log(xtrain + 0.1)
xtest_log = np.log(xtest + 0.1)


# def distance from training set and testing set to training set

train_dist = np.zeros((ytrain.shape[0], ytrain.shape[0]))
test_dist = np.zeros((ytest.shape[0], ytrain.shape[0]))

for i in range(ytrain.shape[0]):
    for j in range(ytrain.shape[0]):
        train_dist[i, j] = np.sum((xtrain_log[i, :] - xtrain_log[j, :])**2)

for i in range(ytest.shape[0]):
    for j in range(ytrain.shape[0]):
        test_dist[i, j] = np.sum((xtest_log[i, :] - xtrain_log[j, :])**2)

# def trainerror

def trainerror(K_):
    label = []
    for i in range(ytrain.shape[0]):
        l = ytrain[np.argpartition(train_dist[i, :], K_)[: K_], :]
        # l = ytrain[np.argsort(train_dist[i, :])[: K_], :]
        if np.sum(l==0) > np.sum(l==1):
            label.append(0)
        else:
            label.append(1)

    error = np.sum(label!=ytrain[:, 0])

    return error / ytrain.shape[0]


# def testerror

def testerror(K_):
    label = []
    for i in range(ytest.shape[0]):
        l = ytrain[np.argpartition(test_dist[i, :], K_)[: K_], :]
        if np.sum(l==0) > np.sum(l==1):
            label.append(0)
        else:
            label.append(1)

    error = np.sum(label!=ytest[:, 0])

    return error / ytest.shape[0]

import matplotlib.pyplot as plt

xaxis = []
train_error = []
test_error = []

K_1 = np.arange(1, 11)
K_2 = np.arange(15, 101, 5)
K = np.concatenate((K_1, K_2), axis=0)

for i in K:
    xaxis.append(i)
    train_error.append(trainerror(i))
    test_error.append(testerror(i))

plt.grid(True)
plt.xlabel(r'$K$')
plt.title('K-Nearest Neighbors')
plt.plot(xaxis, test_error, label = 'test error rates')
plt.plot(xaxis, train_error, label = 'train error rates')
plt.legend()
plt.show()

print(train_error[0], train_error[9], train_error[-1])

print(test_error[0], test_error[9], test_error[-1])