import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TemperedSigmoid(nn.Module):
    def __init__(self, args):
        super(TemperedSigmoid, self).__init__()
        self.scale = args.scale 
        self.temp = args.temp
        self.offset = args.offset

    def forward(self, x):
        return self.scale / (1 + torch.exp(-self.temp * x)) - self.offset

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

        if args.activation == 'relu':
            self.act = nn.ReLU()
        elif args.activation == 'tanh':
            self.act = nn.Tanh()
        elif args.activation == 'tempered':
            self.act = TemperedSigmoid(args)
        else:
            raise ValueError('Unknown activation function: {}'.format(args.activation))

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if args.activation == 'relu':
            self.act = nn.ReLU()
        elif args.activation == 'tanh':
            self.act = nn.Tanh()
        elif args.activation == 'tempered':
            self.act = TemperedSigmoid(args)
        else:
            raise ValueError('Unknown activation function: {}'.format(args.activation))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(self.act(self.conv2(x)))

        x = self.act(self.conv3(x))
        x = self.pool2(self.act(self.conv4(x)))

        x = self.act(self.conv5(x))
        x = self.pool3(self.act(self.conv6(x)))

        x = self.act(self.conv7(x))
        x = self.avgpool(self.conv8(x))
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)


class ScatterLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_groups):
        super(ScatterLinear, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gn = nn.GroupNorm(num_groups, input_dim)

    def forward(self, x):
        x = self.gn(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class ScatterCNN(nn.Module):
    def __init__(self, dataset, input_dim, output_dim, num_groups):
        super(ScatterCNN, self).__init__()
        if dataset == 'mnist' or dataset == 'fmnist':
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 2 * 2, 32)
            self.fc2 = nn.Linear(32, output_dim)
            self.gn = nn.GroupNorm(num_groups, input_dim)
        elif dataset == 'cifar':
            self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 2 * 2, 10)
            self.gn = nn.GroupNorm(num_groups, input_dim)
            self.act = nn.Tanh()
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset))
        self.act = nn.Tanh()
        self.dataset = dataset

    def forward(self, x):
        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            x = self.gn(x)
            x = self.act(self.conv1(x))  
            x = self.pool(x)
            x = self.act(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 32 * 2 * 2) 
            x = self.act(self.fc1(x))
            x = self.fc2(x)
        elif self.dataset == 'cifar':
            x = self.gn(x)
            x = self.act(self.conv1(x))  
            x = self.pool(x)
            x = self.act(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 64 * 2 * 2)
            x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# class ScatterCNN(nn.Module):
#     def __init__(self, dataset, input_dim, output_dim, num_groups):
#         super(ScatterCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=3, stride=2, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(32 * 2 * 2, 32)
#         self.fc2 = nn.Linear(32, output_dim)
#         self.gn = nn.GroupNorm(num_groups, input_dim)
#         self.act = nn.Tanh()

#     def forward(self, x):
#         x = self.gn(x)
#         x = self.act(self.conv1(x))  
#         x = self.pool(x)
#         x = self.act(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 32 * 2 * 2) 
#         x = self.act(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
