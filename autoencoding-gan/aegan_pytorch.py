import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.DCGAN_mnist import Generator, Discriminator, Encoder

mb_size = 32
z_dim = 10
X_dim = 28*28
y_dim = 10
h_dim = 256
cnt = 0
lr = 1e-3
cuda = torch.cuda.is_available()

mnist = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(mnist, batch_size=mb_size, shuffle=True, num_workers=0, drop_last=True)

def BatchIterator():
    while(True):
        for batch_item in data_loader:
            yield batch_item

batch_iterator = BatchIterator()

# Q = Encoder X -> z
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim)
)



G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)


D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
)

Q = Encoder()
G = Generator()
D = Discriminator()

if cuda:
    Q = Q.cuda()
    G = G.cuda()
    D = D.cuda()

def reset_grad():
    Q.zero_grad()
    G.zero_grad()
    D.zero_grad()

Q_solver = optim.Adam(Q.parameters(), lr=lr)
G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)


for it in range(1000000):

    # Reconstruction step
    z = Variable(torch.randn(mb_size, z_dim,1,1))
    if cuda:
        z = z.cuda()
    X_fake = G(z)
    z_recon = Q(X_fake)
    Q_loss = F.mse_loss(z_recon, z)
    Q_loss.backward()
    G_solver.step()
    Q_solver.step()
    reset_grad()

    for _ in range(5):
        # Sample data
        z = Variable(torch.randn(mb_size, z_dim,1,1))
        X, _ = next(batch_iterator)
        X = Variable(X)#.view(-1,X_dim)
        if cuda:
            X = X.cuda()
            z = z.cuda()

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

        D_loss.backward()
        D_solver.step()

        # Weight clipping
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Housekeeping - reset gradient
        reset_grad()

    # Generator forward-loss-backward-update
    X, _ = next(batch_iterator)
    X = Variable(X).view(-1,X_dim)
    z = Variable(torch.randn(mb_size, z_dim,1,1))
    if cuda:
        X = X.cuda()
        z = z.cuda()
    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = -torch.mean(D_fake)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Print and plot every now and then
    if it % 100 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}, Q_loss: {}'
              .format(it, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy(),  Q_loss.cpu().data.numpy()))

        samples = G(z).cpu().data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        torch.save('out/{}.png'.format(str(cnt).zfill(3)))
        cnt += 1
        plt.close(fig)
