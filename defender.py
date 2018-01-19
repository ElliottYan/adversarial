import torch
import math

import torch.autograd as ag
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np

import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import PIL
import matplotlib.pyplot as plt
from dataset import Dataset


class AutoEncoder(nn.Module):
    def __int__(self, ):
        super(AutoEncoder, self).__init__()
        self.input_size = 299 * 299 * 3
        self.c1_sz = 16 * 75 * 75
        self.h1_sz = 16 * 784
        self.h2_sz = 16 * 32
        self.h3_sz = 32
        # unsampling convolution
        self.conv1 = nn.Conv2d(3, 8, 3, stride=(2, 2))
        self.conv2 = nn.Conv2d(8, 16, 3, stride=(2, 2))
        self.conv3 = nn.Conv2d(16, 16, 3, stride=(2, 2))
        self.fc1 = nn.Linear(self.c1_sz, self.h1_sz)
        self.fc2 = nn.Linear(self.h1_sz, self.h2_sz)
        self.fc3 = nn.Linear(self.h2_sz, self.h3_sz)
        self.fc4 = nn.Linear(self.h3_sz, self.h2_sz)
        self.fc5 = nn.Linear(self.h2_sz, self.h1_sz)
        self.fc6 = nn.Linear(self.h1_sz, self.c1_sz)
        self.bi = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU()


    def forward(self, x):
        # flatten the image

        x = self.relu(self.bn(self.conv1(x)))

        x = x.view(x.size(0), -1)

        x = x.view(x.size(0), 28, 28)
        x = self.bi(x)
        # upsampling!

        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn2(self.conv3(c2)))
        # after convolution ---> shape(16 * 75 * 75)

        # encoding
        h1 = F.relu(self.fc1(c3))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        # decoding
        d1 = F.relu(self.fc4(h3))
        d2 = F.relu(self.fc5(d1))
        d3 = F.relu(self.fc6(d2))

        # x =
        return x


class U_net(nn.Module):
    def __init__(self, ):
        super(U_net, self).__init__()
        # input, output, 5*5 conv
        self.conv1 = nn.Conv2d(3, 6, 3, stride=(2, 2))
        self.conv2 = nn.Conv2d(6, 16, 3, stride=(2, 2))

        # self.bi1 = nn.Bilinear()

    def forward(self, x):
        # (2, 2) square pooling kernel.
        x = self.relu(self.conv1(x)), (2, 2)
        x = self.bn1(x)
        x = self.relu(self.cnov2(x)), (2, 2)
        x = self.bn2(x)


        # x = self.sigmoid(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))
        return x


def main():
    img_size = 299
    batch_size = 32
    input_dir = "/home/nisl/adversarial/data"

    dataset = Dataset(input_dir, transform=tf)
    n_classes = len(dataset.classes)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_func = nn.NLLLoss()
