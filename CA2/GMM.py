import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# np.random.seed(0)


image_label = np.random.choice(os.listdir('PIE'), 25, replace=False)
image_label = [int(i) for i in image_label]

print('25 subjects we choosed: {}'.format(image_label))

train_image = []
test_image = []
train_label = []
test_label = []


for i in range(25):
    dir = os.listdir('PIE'+'/'+str(image_label[i]))
    l = len(dir)
    label = np.random.choice(dir, int(l*0.7), replace=False)
    for j in dir:
        img = np.array(Image.open('PIE'+'/'+str(image_label[i])+'/'+ j))
        lab = image_label[i]
        if j in label:
            train_image.append(img)
            train_label.append(lab)
        else:
            test_image.append(img)
            test_label.append(lab)


# my images with label: 0

mytrain_image = []
mytest_image = []
mytrain_label = []
mytest_label = []

mydir = os.listdir('my')
l = len(mydir)
mylabel = np.random.choice(mydir, int(l*0.7), replace=False)

for j in mydir:
    if j in mylabel:
        mytrain_image.append(np.array(Image.open('my'+'/'+j)))
        mytrain_label.append(0)
    else:
        mytest_image.append(np.array(Image.open('my'+'/'+j)))
        mytest_label.append(0)


# np.random.seed(0)


trainset_x = np.concatenate((np.array(train_image).reshape(-1,1024), np.array(mytrain_image).reshape(-1,1024)), axis=0)
trainset_y = np.concatenate((np.array(train_label), np.array(mytrain_label)), axis=0)
testset_x = np.concatenate((np.array(test_image).reshape(-1,1024), np.array(mytest_image).reshape(-1,1024)), axis=0)
testset_y = np.concatenate((np.array(test_label), np.array(mytest_label)), axis=0)



pca_2 = PCA(n_components=2)
pca_2.fit(trainset_x)

X = pca_2.transform(trainset_x)

gmm = GaussianMixture(n_components=3).fit(X)
labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title('GMM visualization for raw face images')
plt.show()

line1 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)
line2 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)
line3 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)

fig, axs= plt.subplots(1, 3)
fig.suptitle('raw face images assigned to the 1st Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line1[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()


fig, axs= plt.subplots(1, 3)
fig.suptitle('raw face images assigned to the 2nd Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line2[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()


fig, axs= plt.subplots(1, 3)
fig.suptitle('raw face images assigned to the 3rd Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line3[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()



# PCA 80

pca = PCA(n_components=80)
pca.fit(trainset_x)

# project to 2d for visualization

pca_2 = PCA(n_components=2)
pca_2.fit(pca.transform(trainset_x))

X = pca_2.transform(pca.transform(trainset_x))

gmm = GaussianMixture(n_components=3).fit(X)
labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title('GMM visualization for PCA 80')
plt.show()


line1 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)
line2 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)
line3 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)

fig, axs= plt.subplots(1, 3)
fig.suptitle('PCA 80 assigned to the 1st Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line1[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()


fig, axs= plt.subplots(1, 3)
fig.suptitle('PCA 80 assigned to the 2nd Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line2[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()


fig, axs= plt.subplots(1, 3)
fig.suptitle('PCA 80 assigned to the 3rd Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line3[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()



# PCA 200

pca = PCA(n_components=200)
pca.fit(trainset_x)

# project to 2d for visualization

pca_2 = PCA(n_components=2)
pca_2.fit(pca.transform(trainset_x))

X = pca_2.transform(pca.transform(trainset_x))

gmm = GaussianMixture(n_components=3).fit(X)
labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title('GMM visualization for PCA 200')
plt.show()


line1 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)
line2 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)
line3 = np.random.choice(trainset_x[labels==0].shape[0], 3, replace=False)

fig, axs= plt.subplots(1, 3)
fig.suptitle('PCA 200 assigned to the 1st Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line1[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()


fig, axs= plt.subplots(1, 3)
fig.suptitle('PCA 200 assigned to the 2nd Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line2[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()


fig, axs= plt.subplots(1, 3)
fig.suptitle('PCA 200 assigned to the 3rd Gaussian component')
for i in range(3):
    axs[i].imshow(trainset_x[line3[i], :].reshape((32, 32)), 'gray')
    axs[i].axis('off')
plt.show()
