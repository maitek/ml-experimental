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
import cv2
import torch.nn.init as init
from models.unet import UNet
from models.dcgan import Discriminator

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Refine GAN')
parser.add_argument('--cuda', action='store_true', default=True)
args = parser.parse_args()

dataset_train = MaterialsDataset("PBR_dataset_256/", test = True)
dataset_test = MaterialsDataset("PBR_dataset_256/", test = False)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=True)

if not torch.cuda.is_available():
    print("Warning cuda not found, falling back on CPU!")
    args.cuda = False

G = UNet(n_channels=3, n_output=3)
D = Discriminator()

if args.cuda:
    G.cuda()
    D.cuda()

lr=0.001

# run forever
for epoch in range(10000):
    lr *= 0.9995
    print("Epoch {}, lr: {}".format(epoch,lr) )
    G_solver = torch.optim.Adam(model.parameters(),lr=lr )
    D_solver = torch.optim.Adam(model.parameters(),lr=lr )

    train_loss = list()
    tic = time()
    # Train Epoch
    for batch_idx, batch_item in enumerate(train_loader):
        X_real = batch_item[0]
        X_fake = batch_item[1]

        if args.cuda:
            X_real, X_fake = X_real.cuda(), X_fake.cuda()

        X_real, X_render = Variable(X_real), Variable(X_fake)
        X_fake = G(X_render)

        # train discriminator
        for _ in range(5):
            D_real = D(X_real)
            D_fake = D(X_fake)

            #D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
            D_loss.backward()
            D_solver.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            reset_grad()

        # train generator
        G_loss = -torch.mean(D_fake)
        G_loss.backward()
        G_solver.step()
        reset_grad()

        train_loss.append(loss.data[0])

        G_loss.backward()
        Q_solver.step()
        reset_grad()

    print('D-loss: {:.4f}, G-loss: {:.4f} time: {:.4f} seconds'.format(D_loss.data.cpu().numpy(),G_loss.data.cpu().numpy(), time()-tic))

    if epoch % 10 == 0:
        torch.save(model,"model_latest.pth")

        if args.cuda:
            X_real, X_fake = X_fake.cpu(), X_real.cpu()

        plt.subplot(321)
        grid = make_grid(X_real.data, nrow=4).numpy()
        grid = np.moveaxis(grid,0,-1)
        plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

        plt.subplot(322)
        grid = make_grid(X_fake.data, nrow=4).numpy()
        grid = np.moveaxis(grid,0,-1)
        plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

        #plt.show()
        out_dir = "output"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig('{}/{}.png'.format(out_dir,str(epoch)), bbox_inches='tight')
        torch.save(model,'{}/checkpoint_{}.pth.tar'.format(out_dir,str(epoch)))
