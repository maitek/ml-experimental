from data_loader import MaterialsDataset

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import numpy as np
import os
from time import time

from itertools import count as forever


dataset = MaterialsDataset("/Users/sundholm/Data/PBR_dataset_cleaned")
test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

#torchvision.transforms.Normalize(mean, std)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 3, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.batch_norm1(self.conv1(x))

        x = F.relu(x)
        x = self.batch_norm2(self.conv2(x))
        x = F.relu(x)
        x = F.tanh(self.conv3(x))

        return x

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# run forever
for epoch in forever():
    print("Epoch {}".format(epoch) )
    for batch_idx, batch_item in enumerate(test_loader):
        tic = time()
        albedo = batch_item["albedo"]
        normal = batch_item["normal"]

        data, target = Variable(albedo), Variable(normal)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Loss: {:.4f} time: {} seconds'.format(loss.data[0], time()-tic))
