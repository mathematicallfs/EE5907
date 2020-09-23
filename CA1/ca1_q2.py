import numpy as np
from scipy import io

data = io.loadmat('spamData.mat')

xtrain = np.array(data['Xtrain'])
xtest = np.array(data['Xtest'])
ytrain = np.array(data['ytrain'])
ytest = np.array(data['ytest'])


def log(x):
    x_log = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_log[i,j] = np.log(x[i,j] + 0.1)
    return x_log

xtrain_log = log(xtrain)
xtest_log = log(xtest)


lambda_ML = np.count_nonzero(ytrain==0)/ytrain.shape[0]



M = np.zeros((57,4))  # likelihood of the feature, returns a matrix with shape (57, 4)
for i in range(57):
    x_i0 = []
    x_i1 = []
    for j in range(ytrain.shape[0]):
        if ytrain[j,0] == 0:
            x_i0.append(xtrain_log[j,i])
        if ytrain[j,0] == 1:
            x_i1.append(xtrain_log[j,i])

    M[i,0] = np.mean(x_i0) 
    M[i,1] = np.var(x_i0)
    M[i,2] = np.mean(x_i1)
    M[i,3] = np.var(x_i1)


Ntest_error = 0
for i in range(xtest.shape[0]):
    ptest_0 = 0
    ptest_1 = 0

    for j in range(57):
        ptest_0 = ptest_0 - 1/2 * np.log(2 * np.pi * M[j,1]) - (xtest_log[i,j] - M[j,0])*(xtest_log[i,j] - M[j,0]) /(2 * M[j,1])
        ptest_1 = ptest_1 - 1/2 * np.log(2 * np.pi * M[j,3]) - (xtest_log[i,j] - M[j,2])*(xtest_log[i,j] - M[j,2]) /(2 * M[j,3])
        
    ptest_0 = ptest_0 + np.log(lambda_ML)   # lop p(y=0|D)
    ptest_1 = ptest_1 + np.log(1-lambda_ML)  # lop p(y=1|D)
        
    if (ptest_0 > ptest_1 and ytest[i,0]==1) or (ptest_0 < ptest_1 and ytest[i,0]==0):
        Ntest_error = Ntest_error + 1
        
print(Ntest_error / ytest.shape[0])


Ntrain_error = 0
for i in range(xtrain.shape[0]):
    ptrain_0 = 0
    ptrain_1 = 0

    for j in range(57):
        ptrain_0 = ptrain_0 - 1/2 * np.log(2 * np.pi * M[j,1]) - (xtrain_log[i,j] - M[j,0])*(xtrain_log[i,j] - M[j,0])/(2 * M[j,1])
        ptrain_1 = ptrain_1 - 1/2 * np.log(2 * np.pi * M[j,3]) - (xtrain_log[i,j] - M[j,2])*(xtrain_log[i,j] - M[j,2])/(2 * M[j,3])
        
    ptrain_0 = ptrain_0 + np.log(lambda_ML)   # lop p(y=0|D)
    ptrain_1 = ptrain_1 + np.log(1-lambda_ML)  # lop p(y=1|D)
        
    if (ptrain_0 > ptrain_1 and ytrain[i,0]==1) or (ptrain_0 < ptrain_1 and ytrain[i,0]==0):
        Ntrain_error = Ntrain_error + 1
        
print(Ntrain_error / ytrain.shape[0])