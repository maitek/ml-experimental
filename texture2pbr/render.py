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
import torch.nn.init as init
from unet import UNet

dataset = MaterialsDataset("PBR_dataset/")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = UNet(n_channels=3, n_output=5)
if os.path.exists("model_latest.pth"):
    model = torch.load("model_latest.pth")
model = model.cuda()

for batch in data_loader:
    albedo = batch["albedo"]
    albedo = albedo.cuda()
    data = Variable(albedo)
    output = model(data)

    import pdb; pdb.set_trace()

    for i in range(0,11):
        output_normal = output[i,0:3,:,:].cpu().data.numpy()
        output_normal = np.moveaxis(output_normal,0,-1)

        output_ao = output[i,2:3,:,:].cpu().data.numpy()
        output_ao = np.moveaxis(output_ao,0,-1)

        output_roughness = output[i,3:4,:,:].cpu().data.numpy()
        output_roughness = np.moveaxis(output_roughness,0,-1)
        import pdb; pdb.set_trace()
