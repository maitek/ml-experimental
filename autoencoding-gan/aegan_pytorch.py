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
mb_size = 96
z_dim = 5
X_dim = 28*28
h_dim = 64
lr = 1e-4
cnt=0
out_dir = "out_aegan"

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=mb_size, shuffle=True, num_workers=0, drop_last=True)



# Q = Encoder X -> z
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim)
)

# G = Generator z -> X
G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

# D= Discriminator X_real vs X_fake
D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1)
    #torch.nn.Sigmoid()
)

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

    for batch_idx, batch_item in enumerate(data_loader):
        
        z = Variable(torch.randn(mb_size, z_dim))

        if cuda:
            X = X.cuda()
            z = z.cuda()

        # Reconstruction step
        X_fake = G(z)
        z_recon = Q(X_fake)

        recon_loss = F.mse_loss(z_recon, z)
        recon_loss.backward()
        #G_solver.step()
        Q_solver.step()
        reset_grad()

        # Discriminator step
        for _ in range(5):
            z = Variable(torch.randn(mb_size, z_dim))
            X = Variable(batch_item[0]).view(-1,X_dim)
            X_fake = G(z)
            D_real = D(X)
            D_fake = D(X_fake)

            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
            D_loss.backward()
            D_solver.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            D_solver.step()
            reset_grad()

        # Generator step
        X_fake = G(z)
        D_fake = D(X_fake)
        #G_loss = -torch.mean(torch.log(D_fake))
        G_loss = -torch.mean(D_fake)
        G_loss.backward()
        G_solver.step()
        reset_grad()

        if batch_idx % 100 == 0:
            print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
                  .format(batch_idx, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))

        # Print and plot every now and then
        if batch_idx % 100 == 0:
            samples = G(z)
            samples = samples.view(-1,1,28,28)

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
                plt.imshow(sample[:,:,0],cmap='Greys_r')
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            plt.savefig('{}/{}.png'
                        .format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')
            cnt += 1
            plt.close(fig)
