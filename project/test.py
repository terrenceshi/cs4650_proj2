import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim

from net import Net

import os

def main():
    #parameters:
    numEpochs = 10
    batchSize = 4
    learningRate = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #data:
    testLen = sum([len(files) for r, d, files in os.walk("Data/C50test")])
    print('test:', testLen)

    trainLen = sum([len(files) for r, d, files in os.walk("Data/C50train")])
    print('train:', trainLen)

    trainData = datasets.ImageFolder('Data/C50train')
    testData = datasets.ImageFolder('Data/C50test')

    testData, validData = torch.utils.data.random_split(testData, [testLen/2, testLen/2]) #split test data into 2 for validation set

    trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle=True)
    validLoader = DataLoader(validData, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testData, batch_size = batchSize, shuffle=True)

    # get list of authors
    classes = []
    rootdir = 'Data/C50train'
    for it in os.scandir(rootdir): #scan subdirectory and append each element to list of classes
        if it.is_dir():
            classes.append(it.path.replace(rootdir + "\\" , '')) #remove 'C50train\' from string

    #insert model here, nothing is in the model yet
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=learningRate, momentum=0.9)

    #start training

    for e in range(numEpochs):
        #training loop
        train_loss = 0.0
        net.train()
        for data, labels in trainLoader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            target = net(data)

            loss = criterion(target, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        #validation loop
        valid_loss = 0.0
        net.eval()  # Optional when not using Model Specific layer
        for data, labels in validLoader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            target = net(data)
            loss = criterion(target, labels)
            valid_loss = loss.item() * data.size(0)

        print(
            f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(trainLoader)} \t\t Validation Loss: {valid_loss / len(validLoader)}')

        #save model if validation loss decreases
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), 'Models/saved_model.pth')

    #test loop
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


if __name__ == "__main__":
    main()