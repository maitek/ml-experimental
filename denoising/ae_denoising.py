import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import PIL

torch.manual_seed(1)    # reproducible
noise_image = np.asarray(PIL.Image.open("example.png"))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        nc = 3
        ndf = 1

        self.encoder_layers = [
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()]
        for idx, module in enumerate(self.encoder_layers):
            self.add_module(str(idx), module)

        self.decoder_layers = [
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.Sigmoid(),
        ]

    def forward(self, x):

        # encoder

        for layer in self.encoder_layers:
            x = layer(x)
            #print(x.size(), layer)
        #import pdb; pdb.set_trace()
        #for layer in self.decoder_layers:
        #    x = layer(x)
        #    print(x.size())
        return x





x = np.expand_dims(np.swapaxes(noise_image,0,2),0).astype(np.float32)/255

x = Variable(torch.from_numpy(x))
model = AutoEncoder()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.MSELoss()

for idx in range(10000):
    out = model(x)
    # mean square error
    loss = loss_func(out, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if idx % 10 == 0:
        print(loss.data[0])
    if idx % 100 == 0:
        image = out.data.numpy()[0,:,:,:]
        image = np.swapaxes(image,0,2)
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(noise_image)
        plt.show()
