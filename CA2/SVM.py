import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


np.random.seed(0)


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


np.random.seed(0)


trainset_x = np.concatenate((np.array(train_image).reshape(-1,1024), np.array(mytrain_image).reshape(-1,1024)), axis=0)
trainset_y = np.concatenate((np.array(train_label), np.array(mytrain_label)), axis=0)
testset_x = np.concatenate((np.array(test_image).reshape(-1,1024), np.array(mytest_image).reshape(-1,1024)), axis=0)
testset_y = np.concatenate((np.array(test_label), np.array(mytest_label)), axis=0)


clf_001 = LinearSVC(C=0.01, max_iter=1000)  # change 1000 to 10000 for convergence
clf_001.fit(trainset_x, trainset_y)
accuracy_001 = clf_001.score(testset_x, testset_y)


clf_01 = LinearSVC(C=0.1, max_iter=1000)
clf_01.fit(trainset_x, trainset_y)
accuracy_01 = clf_01.score(testset_x, testset_y)


clf_1 = LinearSVC(C=1, max_iter=1000)
clf_1.fit(trainset_x, trainset_y)
accuracy_1 = clf_1.score(testset_x, testset_y)


print('Classification accuracy on raw face images for C = 0.01: {:.2f}%'.format(100*accuracy_001))
print('Classification accuracy on raw face images for C = 0.1: {:.2f}%'.format(100*accuracy_01))
print('Classification accuracy on raw face images for C = 1: {:.2f}%'.format(100*accuracy_1))


# PCA 80

pca = PCA(n_components=80)
pca.fit(trainset_x)

clf_001 = LinearSVC(C=0.01, max_iter=1000)
clf_001.fit(pca.transform(trainset_x), trainset_y)
accuracy_001 = clf_001.score(pca.transform(testset_x), testset_y)

clf_01 = LinearSVC(C=0.1, max_iter=1000)
clf_01.fit(pca.transform(trainset_x), trainset_y)
accuracy_01 = clf_01.score(pca.transform(testset_x), testset_y)


clf_1 = LinearSVC(C=1, max_iter=1000)
clf_1.fit(pca.transform(trainset_x), trainset_y)
accuracy_1 = clf_1.score(pca.transform(testset_x), testset_y)


print('Classification accuracy on PCA 80 for C = 0.01: {:.2f}%'.format(100*accuracy_001))
print('Classification accuracy on PCA 80 for C = 0.1: {:.2f}%'.format(100*accuracy_01))
print('Classification accuracy on PCA 80 for C = 1: {:.2f}%'.format(100*accuracy_1))




# PCA 200

pca = PCA(n_components=200)
pca.fit(trainset_x)

clf_001 = LinearSVC(C=0.01, max_iter=1000)
clf_001.fit(pca.transform(trainset_x), trainset_y)
accuracy_001 = clf_001.score(pca.transform(testset_x), testset_y)

clf_01 = LinearSVC(C=0.1, max_iter=1000)
clf_01.fit(pca.transform(trainset_x), trainset_y)
accuracy_01 = clf_01.score(pca.transform(testset_x), testset_y)


clf_1 = LinearSVC(C=1, max_iter=1000)
clf_1.fit(pca.transform(trainset_x), trainset_y)
accuracy_1 = clf_1.score(pca.transform(testset_x), testset_y)


print('Classification accuracy on PCA 200 for C = 0.01: {:.2f}%'.format(100*accuracy_001))
print('Classification accuracy on PCA 200 for C = 0.1: {:.2f}%'.format(100*accuracy_01))
print('Classification accuracy on PCA 200 for C = 1: {:.2f}%'.format(100*accuracy_1))



