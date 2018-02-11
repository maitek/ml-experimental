import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

cuda = False
cnt = 0
lr = 1e-4
out_dir = "out"
batch_size = 96

nc = 3 # number of channels
nz = 64 # size of latent vector
ngf = 1 # decoder (generator) filter factor
ndf = 1 # encoder filter factor
h_dim = 128 # discriminator hidden size
lam = 10 # regulization coefficient


transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Scale(64),
        transforms.ToTensor(),
         #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

dataset = datasets.ImageFolder('/Users/sundholm/Data/celeba', transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
#import pdb; pdb.set_trace()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False),
        ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = [
            nn.Linear(nz, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

Q = Encoder()
G = Decoder()
D = Discriminator()

if cuda:
    Q = Q.cuda()
    G = G.cuda()
    D = D.cuda()

QG_solver = optim.Adam(Q.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)

def zero_grads():
    G.zero_grad()
    Q.zero_grad()
    D.zero_grad()

for it in range(1000000):

    for batch_idx, batch_item in enumerate(data_loader):
        X = Variable(batch_item[0])
        if cuda:
            X = X.cuda()

        # Update disciriminator
        z = Q(X)
        z_sample = Variable(torch.randn(batch_size, nz))
        D_real = D(z_sample)
        D_fake = D(z.view(batch_size,-1))

        D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
        loss = lam * D_loss
        #loss = -(torch.mean(D_real) - torch.mean(D_fake))
        loss.backward()
        D_solver.step()
        zero_grads()

        # Updated encoder and decoder
        z = Q(X)
        D_fake = D(z.view(batch_size,-1))
        X_recon = G(z)

        G_loss = -torch.mean(torch.log(D_fake))
        recon_loss = F.mse_loss(X_recon, X)
        loss =  recon_loss + lam * G_loss
        loss.backward()
        QG_solver.step()
        zero_grads()

        if it % 1 == 0:
            print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
                  .format(it, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))

        # Print and plot every now and then
        if it % 10 == 0:

            z_sample = z_sample.unsqueeze(2).unsqueeze(3) # add 2 dimensions

            # Generate sample images

            samples = G(z_sample)

            if cuda:
                samples = samples.cpu()
            samples = samples.data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                sample = np.swapaxes(sample,0,2)
                plt.imshow(sample)


            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            plt.savefig('{}/{}.png'
                        .format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')
            cnt += 1
            plt.close(fig)
