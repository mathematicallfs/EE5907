import numpy as np
from scipy import io
from scipy.special import expit
import matplotlib.pyplot as plt

data = io.loadmat('spamData.mat')

xtrain = np.array(data['Xtrain'])
xtest = np.array(data['Xtest'])
ytrain = np.array(data['ytrain'])
ytest = np.array(data['ytest'])


xtrain_log = np.log(xtrain + 0.1)
xtest_log = np.log(xtest + 0.1)

# concatenate 1 to strart of x_i

a = np.ones((xtrain_log.shape[0], 1))
b = np.ones((xtest_log.shape[0], 1))

xtrain_concat = np.concatenate((a, xtrain_log), axis = 1)
xtest_concat = np.concatenate((b, xtest_log), axis = 1)


# define gradient 

def grad(w_):
    mu = expit(xtrain_concat.dot(w_)) 
    grad = xtrain_concat.T.dot(mu - ytrain)

    return grad

# define hessian

def hessian(w_):
    mu = expit(xtrain_concat.dot(w_))
    S = np.diag(np.squeeze((mu * (1 - mu)), axis=1))
    h = xtrain_concat.T.dot(S).dot(xtrain_concat)

    return h

# def trainerror

def trainerror(w_):
    mu = expit(xtrain_concat.dot(w_))
    delta = np.abs(mu - ytrain) - 0.5 * np.ones((ytrain.shape))
    error = np.sum(delta>0)

    return error / ytrain.shape[0]

# def testerror

def testerror(w_):
    mu = expit(xtest_concat.dot(w_))
    delta = np.abs(mu - ytest) - 0.5 * np.ones((ytest.shape))
    error = np.sum(delta>0)

    return error / ytest.shape[0]

# training 

lam_1 = np.arange(1, 11)
lam_2 = np.arange(15, 101, 5)
lam = np.concatenate((lam_1, lam_2), axis=0)

xaxis = []
train_error = []
test_error = []

I = np.identity(58)
I[0,0] = 0


for j in lam:
    xaxis.append(j)
    eps = 1
    w = np.zeros((58, 1))
    
    while eps > 1e-3:
        h = hessian(w) + j * I
        g = grad(w) + j * np.concatenate((np.zeros((1, 1)), w[1:]), axis=0)
        w = w - (np.linalg.inv(h)).dot(g)
        eps = np.sum(np.abs((np.linalg.inv(h)).dot(g)))

    train_error.append(trainerror(w))
    test_error.append(testerror(w))

plt.grid(True)
plt.xlabel(r'$\lambda$')
plt.title('Logistic regression')
plt.plot(xaxis, test_error, label = 'test error rates')
plt.plot(xaxis, train_error, label = 'train error rates')
plt.legend()
plt.show()

print(train_error[0], train_error[9], train_error[-1])

print(test_error[0], test_error[9], test_error[-1])
