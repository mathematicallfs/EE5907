import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(1)


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

rand = np.random.choice(range(len(train_image)), 500, replace=False)
visualize_x = np.array(train_image)[rand].reshape(-1,1024)
visualize_x = np.concatenate((visualize_x, np.array(mytrain_image).reshape(-1,1024)), axis=0)
visualize_y = np.array(train_label)[rand]
visualize_y = np.concatenate((visualize_y, np.array(mytrain_label)), axis=0)


lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(visualize_x, visualize_y)

projected = lda.transform(visualize_x)
plt.scatter(projected[:, 0], projected[:, 1], c=visualize_y)
plt.plot(projected[visualize_y==0][:, 0], projected[visualize_y==0][:, 1], 'r*', markersize=15, label='my photos')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.title('LDA for 2d projected data visualization')
plt.legend()
plt.show()


lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(visualize_x, visualize_y)

projected = lda.transform(visualize_x)
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c=visualize_y)
plt.plot(projected[visualize_y==0][:, 0], projected[visualize_y==0][:, 1], projected[visualize_y==0][:, 2], 'r*', markersize=15, label='my photos')
plt.title('LDA for 3d projected data visualization')
plt.legend()
plt.show()



trainset_x = np.concatenate((np.array(train_image).reshape(-1,1024), np.array(mytrain_image).reshape(-1,1024)), axis=0)
trainset_y = np.concatenate((np.array(train_label), np.array(mytrain_label)), axis=0)
testset_x = np.concatenate((np.array(test_image).reshape(-1,1024), np.array(mytest_image).reshape(-1,1024)), axis=0)
testset_y = np.concatenate((np.array(test_label), np.array(mytest_label)), axis=0)

# LDA 2

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(trainset_x, trainset_y)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(lda.transform(trainset_x), trainset_y)

accuracy_cmu = accuracy_score(np.array(test_label), knn.predict(lda.transform(np.array(test_image).reshape(-1,1024))))
accuracy_my = accuracy_score(np.array(mytest_label), knn.predict(lda.transform(np.array(mytest_image).reshape(-1,1024))))
print("Classification accuracy for LDA 2 on CMU PIE test images: {:.2f}%, Classification accuracy for LDA 2 on my own photos: {:.2f}%".format(100*accuracy_cmu, 100*accuracy_my))

# LDA 3

lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(trainset_x, trainset_y)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(lda.transform(trainset_x), trainset_y)

accuracy_cmu = accuracy_score(np.array(test_label), knn.predict(lda.transform(np.array(test_image).reshape(-1,1024))))
accuracy_my = accuracy_score(np.array(mytest_label), knn.predict(lda.transform(np.array(mytest_image).reshape(-1,1024))))
print("Classification accuracy for LDA 3 on CMU PIE test images: {:.2f}%, Classification accuracy for LDA 3 on my own photos: {:.2f}%".format(100*accuracy_cmu, 100*accuracy_my))



# LDA 9

lda = LinearDiscriminantAnalysis(n_components=9)
lda.fit(trainset_x, trainset_y)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(lda.transform(trainset_x), trainset_y)

accuracy_cmu = accuracy_score(np.array(test_label), knn.predict(lda.transform(np.array(test_image).reshape(-1,1024))))
accuracy_my = accuracy_score(np.array(mytest_label), knn.predict(lda.transform(np.array(mytest_image).reshape(-1,1024))))
print("Classification accuracy for LDA 9 on CMU PIE test images: {:.2f}%, Classification accuracy for LDA 9 on my own photos: {:.2f}%".format(100*accuracy_cmu, 100*accuracy_my))



