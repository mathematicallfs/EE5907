import numpy as np
from scipy import io
from tqdm import tqdm
import matplotlib.pyplot as plt

data = io.loadmat('spamData.mat')

xtrain = np.array(data['Xtrain'])
xtest = np.array(data['Xtest'])
ytrain = np.array(data['ytrain'])
ytest = np.array(data['ytest'])


def binarization(x):
    x_bi = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] > 0:
                x_bi[i,j] = 1
            else:
                x_bi[i,j] = 0
    return x_bi

xtrain_bi = binarization(xtrain)
xtest_bi = binarization(xtest)


lambda_ML = np.count_nonzero(ytrain==0)/ytrain.shape[0]


def feature_likelihood(alpha):  # likelihood of the feature, returns a matrix with shape (57, 4)
    
    a = np.zeros((57,4))
    for i in range(57):
        check_sum = xtrain_bi[:, i] + ytrain[:, 0]
        N_00 = np.sum(check_sum==0)
        N_11 = np.sum(check_sum==2)
        check_minus = xtrain_bi[:, i] - ytrain[:, 0]
        N_01 = np.sum(check_minus==-1)
        N_10 = np.sum(check_minus==1)

        a[i,0] = (N_00 + alpha) / (np.sum(ytrain[:, 0]==0) + 2 * alpha)
        a[i,1] = (N_01 + alpha) / (np.sum(ytrain[:, 0]==1) + 2 * alpha)
        a[i,2] = (N_10 + alpha) / (np.sum(ytrain[:, 0]==0) + 2 * alpha)
        a[i,3] = (N_11 + alpha) / (np.sum(ytrain[:, 0]==1) + 2 * alpha)
    
    return a


def test_error(alpha):
    M = feature_likelihood(alpha)
    N_error = 0
    for i in range(xtest.shape[0]):
        p_0 = 0
        p_1 = 0

        for j in range(57):
            if xtest_bi[i,j] == 0.:
                p_0 = p_0 + np.log(M[j,0])
                p_1 = p_1 + np.log(M[j,1])
            if xtest_bi[i,j] == 1.:
                p_0 = p_0 + np.log(M[j,2])
                p_1 = p_1 + np.log(M[j,3])
        
        p_0 = p_0 + np.log(lambda_ML)   # lop p(y=0|D)
        p_1 = p_1 + np.log(1-lambda_ML)  # lop p(y=1|D)
        
        if (p_0 > p_1 and ytest[i,0]==1) or (p_0 < p_1 and ytest[i,0]==0):
            N_error = N_error + 1
        
    return N_error / ytest.shape[0]


def train_error(alpha):
    M = feature_likelihood(alpha)
    N_error = 0
    for i in range(xtrain.shape[0]):
        p_0 = 0
        p_1 = 0

        for j in range(57):
            if xtrain_bi[i,j] == 0.:
                p_0 = p_0 + np.log(M[j,0])
                p_1 = p_1 + np.log(M[j,1])
            if xtrain_bi[i,j] == 1.:
                p_0 = p_0 + np.log(M[j,2])
                p_1 = p_1 + np.log(M[j,3])
        
        p_0 = p_0 + np.log(lambda_ML)   # lop p(y=0|D)
        p_1 = p_1 + np.log(1-lambda_ML)  # lop p(y=1|D)
        
        if (p_0 > p_1 and ytrain[i,0]==1) or (p_0 < p_1 and ytrain[i,0]==0):
            N_error = N_error + 1
        
    return N_error / ytrain.shape[0]


alpha = np.arange(0,100.5,0.5)
testerror = []
trainerror = []

for i in tqdm(alpha):
    testerror.append(test_error(i))
    trainerror.append(train_error(i))

plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.title('Beta-binomial Naive Bayes')
plt.plot(alpha, testerror, label = 'test error rates')
plt.plot(alpha, trainerror, label = 'train error rates')
plt.legend(loc='best')
plt.show()

print(train_error(1), train_error(10), train_error(100))

print(test_error(1), test_error(10), test_error(100))
