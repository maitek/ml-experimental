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
from torchvision.utils import make_grid
from itertools import count as forever


dataset_train = MaterialsDataset("/Users/sundholm/Data/PBR_dataset_cleaned", test = False)
dataset_test = MaterialsDataset("/Users/sundholm/Data/PBR_dataset_cleaned", test = True)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=True)

#torchvision.transforms.Normalize(mean, std)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 2, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.batch_norm1(self.conv1(x))

        x = F.relu(x)
        x = self.batch_norm2(self.conv2(x))
        x = F.relu(x)
        x = F.sigmoid(self.conv3(x))

        return x

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# run forever
for epoch in forever():
    print("Epoch {}".format(epoch) )
    train_loss = list()
    tic = time()
    # Train Epoch
    for batch_idx, batch_item in enumerate(train_loader):
        albedo = batch_item["albedo"]
        normal = batch_item["normal"]

        data, target = Variable(albedo), Variable(normal)
        optimizer.zero_grad()
        output = model(data)
        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])

    # Test Epoch
    test_loss = list()
    for batch_idx, batch_item in enumerate(test_loader):
        albedo = batch_item["albedo"]
        normal = batch_item["normal"]
        data, target = Variable(albedo), Variable(normal)
        output = model(data)
        loss = F.l1_loss(output, target)
        test_loss.append(loss.data[0])

    print('Train loss: {:.4f}, Test loss: {:.4f} time: {:.4f} seconds'.format(np.mean(train_loss),np.mean(test_loss), time()-tic))
    if epoch % 10 == 0:
        normal_grid = make_grid(output.data[0], nrow=4).numpy()
        normal_grid = np.moveaxis(normal_grid,0,-1)
        normal_grid = np.dstack((normal_grid, np.zeros_like(normal_grid[:,:,1])))

        plt.subplot(121)
        plt.imshow(normal_grid)

        normal_grid_true = make_grid(data.data[0], nrow=4).numpy()
        normal_grid_true = np.moveaxis(normal_grid_true,0,-1)
        normal_grid_true = np.dstack((normal_grid_true, np.zeros_like(normal_grid_true[:,:,1])))
        import pdb; pdb.set_trace()
        plt.subplot(122)
        plt.imshow(normal_grid_true)
        plt.show()
