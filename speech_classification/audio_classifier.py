from dataloader import AudioDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

dataset_train = AudioDataset("/Users/sundholm/Data/kaggle_tf_sound/train/audio", train=True)
dataset_test = AudioDataset("/Users/sundholm/Data/kaggle_tf_sound/train/audio", train=False)
# idea. train network to segment silence using noisy labels
# add likelyhood of voice in segment as an extra output
use_cuda = False

class AudioClassifier(nn.Module):
    def __init__(self,num_classes, batch_size=1):
        super(AudioClassifier, self).__init__()

        self.hsize = 64
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.conv_layers = [
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1, dilation=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, dilation=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1), nn.LeakyReLU(0.2, inplace=True)]

        for idx, module in enumerate(self.conv_layers):
            self.add_module(str(idx), module)

        # recurrent layers
        self.rnn = nn.GRUCell(64, self.hsize)

        # classifier output
        self.dense = nn.Linear(self.hsize, self.num_classes, bias=True)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):

        # Feature extractor
        for layer in self.conv_layers:
            #print(layer,x.size())
            x = layer(x)

        hx = Variable(torch.randn(self.batch_size, self.hsize))

        # RNN over third dim of x
        for i in range(x.size()[2]):
            hx = self.rnn(x[:,:,0], hx)

        # classifier
        x = self.dense(hx)
        return self.log_softmax(x)

batch_size = 10

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
model = AudioClassifier(num_classes=dataset_train.num_classes)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(1000):
    for batch_idx, batch_item in enumerate(train_loader):
        x = batch_item["wave_sequence"]
        y = batch_item["class_idx"]

        if use_cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        optimizer.zero_grad()
        output = model(x)

        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

        if batch_idx % 100:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
