import cv2
import torch.nn as nn
from matplotlib import pyplot as plt
import torch
import torch.optim as toptim
from dataset import SignDataset
from network import NetConv
import sklearn.metrics as metrics
import numpy as np
import os


def train(model, device, criterion, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, criterion, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = np.zeros((8,8)) # initialize confusion matrix
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                          pred.cpu(), target.cpu(), labels=[0,1,2,3,4,5,6,7])
        np.set_printoptions(precision=4, suppress=True)
        print(type(conf_matrix))
        print(conf_matrix)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    #gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #load data
    dataset = SignDataset("../../../za_traffic_2020/traffic_train/images/", "../../../za_traffic_2020/traffic_train/train_traffic_sign_dataset.json")
    train_set, test_set = torch.utils.data.random_split(dataset, [10000, 1010])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    #train
    net = NetConv().to(device)
    criterion = nn.NLLLoss()
    optimizer = toptim.Adam(net.parameters(), lr=1e-3) 
    for epoch in range(20):
        train(net, device, criterion, train_loader, optimizer, epoch)
        test(net, device, criterion, test_loader)


if __name__ == "__main__":
    
    main()