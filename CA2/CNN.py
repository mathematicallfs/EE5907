import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import matplotlib.pyplot as plt

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


torchtrainset_x = torch.tensor(trainset_x/255).float().to("cuda")
torchtestset_x = torch.tensor(testset_x/255).float().to("cuda")
torchtrainset_y = torch.zeros((trainset_y.shape[0],)).int().long().to("cuda")
torchtestset_y = torch.zeros((testset_y.shape[0])).int().long().to("cuda")


# rescale labels to [0, 25]


for i in range(trainset_y.shape[0]-2, -1, -1):
    if trainset_y[i] == trainset_y[i+1]:
        torchtrainset_y[i] = torchtrainset_y[i+1]
    else:
        torchtrainset_y[i] = torchtrainset_y[i+1] + 1


for i in range(testset_y.shape[0]-2, -1, -1):
    if testset_y[i] == testset_y[i+1]:
        torchtestset_y[i] = torchtestset_y[i+1]
    else:
        torchtestset_y[i] = torchtestset_y[i+1] + 1


# create dataloader

class subDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index].reshape((1, 32, 32))
        label = self.Label[index]
        return data, label


traindataset = subDataset(torchtrainset_x, torchtrainset_y)
testdataset = subDataset(torchtestset_x, torchtestset_y)
trainloader = DataLoader.DataLoader(traindataset, batch_size=32, shuffle=True)
testloader = DataLoader.DataLoader(testdataset, batch_size=10000, shuffle=False)


# Define a Convolution Neural Network


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1250, 500)
        self.fc2 = nn.Linear(500, 26)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return  x


cnn = CNN().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

x = []
y_1 = []
y_2 = []

for epoch in range(101):  # loop over the dataset multiple times

    train_loss = 0.0
    test_loss = 0.0
    x.append(epoch)
    for i, traindata in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        traininputs, trainlabels = traindata

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(traininputs)
        trainloss = criterion(outputs, trainlabels)
        trainloss.backward()
        optimizer.step()

        # print statistics
        train_loss += trainloss.item()

    y_1.append(train_loss / len(trainloader))

    with torch.no_grad():
        for i, testdata in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            testinputs, testlabels = testdata

            # forward + backward + optimize
            outputs = cnn(testinputs)
            testloss = criterion(outputs, testlabels)

            # print statistics
            test_loss += testloss.item()

        y_2.append(test_loss / len(testloader))

    if epoch % 20 == 0:
        print('Epoch: {}, train_loss: {}'.format(epoch, train_loss / len(trainloader)))
        print('Epoch: {}, test_loss: {}'.format(epoch, test_loss / len(testloader)))

print('Finished Training')

plt.plot(x, y_1, label='Training loss')
plt.plot(x, y_2, label='Testing loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Evolution of the training and testing loss")
plt.grid()
plt.show()

# Final Accuracy on train set and test set

def dataset_accuracy(data_loader, name=""):
    correct = 0
    total = 0
    for images, labels in data_loader:
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * float(correct) / total
    print('Final accuracy of the network on the {} {} images: {:.2f}%'.format(total, name, accuracy))


dataset_accuracy(trainloader, "train")
dataset_accuracy(testloader, "test")
