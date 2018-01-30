from data_loader import MaterialsDataset
import argparse
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
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Texture to PBR')
parser.add_argument('--cuda', action='store_true', default=True)
args = parser.parse_args()

dataset_train = MaterialsDataset("PBR_dataset_256/", test = True)
dataset_test = MaterialsDataset("PBR_dataset_256/", test = False)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=True)

#torchvision.transforms.Normalize(mean, std)

if not torch.cuda.is_available():
    print("Warning cuda not found, falling back on CPU!")
    args.cuda = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.net = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 3*4**2, kernel_size=3, padding=1), nn.ELU(),
            nn.PixelShuffle(4),
            nn.Sigmoid(),
            #nn.Upsample(scale_factor=2),
            #nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ELU(),
            #nn.Upsample(scale_factor=2),
            #nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ELU(),

            #nn.Conv2d(32, 3, kernel_size=3, padding=1),

        ]

        for idx, module in enumerate(self.net):
            self.add_module(str(idx), module)

    def forward(self, x):

        for layer in self.net:
            #print(x.size())
            x = layer(x)
        return x

"""
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3* upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
init.orthogonal(self.conv4.weight)
"""

model = Net()
if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# run forever
for epoch in forever():
    print("Epoch {}".format(epoch) )
    train_loss = list()
    tic = time()
    # Train Epoch
    for batch_idx, batch_item in enumerate(train_loader):
        albedo = batch_item["albedo"]
        normal = batch_item["normal"]

        if args.cuda:
            albedo, normal = albedo.cuda(), normal.cuda()
        data, target = Variable(albedo), Variable(normal)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])

    # Test Epoch
    test_loss = list()
    for batch_idx, batch_item in enumerate(test_loader):
        albedo = batch_item["albedo"]
        normal = batch_item["normal"]
        if args.cuda:
            albedo, normal = albedo.cuda(), normal.cuda()
        data, target = Variable(albedo), Variable(normal)
        output = model(data)
        loss = F.mse_loss(output, target)
        test_loss.append(loss.data[0])

    print('Train loss: {:.4f}, Test loss: {:.4f} time: {:.4f} seconds'.format(np.mean(train_loss),np.mean(test_loss), time()-tic))
    if epoch % 100 == 0:

        if args.cuda:
            output, normal = output.cpu(), normal.cpu()

        # pad normal map
        nb, c, h, w = output.size()

        #import pdb; pdb.set_trace()

        #output = torch.cat((output, torch.zeros(nb,1,h,w)), 1)

        normal_grid = make_grid(output.data, nrow=4).numpy()
        normal_grid = np.moveaxis(normal_grid,0,-1)
        #normal_grid = np.dstack((normal_grid, np.zeros_like(normal_grid[:,:,1])))

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(normal_grid, cv2.COLOR_BGR2RGB))
        #import pdb; pdb.set_trace()
        #normal = torch.cat((normal, torch.zeros(nb,1,h,w)), 1)
        normal_grid_true = make_grid(normal, nrow=4).numpy()
        normal_grid_true = np.moveaxis(normal_grid_true,0,-1)
        #normal_grid_true = np.dstack((normal_grid_true, np.zeros_like(normal_grid_true[:,:,1])))

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(normal_grid_true, cv2.COLOR_BGR2RGB))
        #plt.show()
        out_dir = "output"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig('{}/{}.png'.format(out_dir,str(epoch)), bbox_inches='tight')
        torch.save('{}/checkpoint_{}.pth.tar'.format(out_dir,str(epoch)))
