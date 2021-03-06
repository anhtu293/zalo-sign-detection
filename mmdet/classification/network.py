import torch.nn as nn


class NetConv(nn.Module):

    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 40, kernel_size=5)
        self.maxpool = nn.MaxPool2d(3, 1)
        self.fc = nn.Linear(116*116*40, 8)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.log_softmax(x)

        return x