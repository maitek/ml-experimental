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


#data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
blender_folder = "Blender_PBR_Tester"

model = UNet(n_channels=3, n_output=5)
if os.path.exists("model_latest.pth"):
    model = torch.load("model_latest.pth")
model = model.cuda()


input_folder = "PBR_dataset/"
texture_folders = os.listdir("PBR_dataset/")
#textures = os.listdir("stone_texture_1024/")

def string_contains(string, sub_strings):
    for sub in sub_strings:
        if sub in string.lower():
            return True
    else:
        return False

cnt = 1

for texture_folder in texture_folders:
#for texture in texture:
    path = os.path.join(input_folder,texture_folder)
    files = os.listdir(path)
    files = [x for x in files if x.endswith(".png")]

    for file_name in files:
        if string_contains(file_name, ["color", "albedo", "alb"]):
            albedo_in = cv2.imread(os.path.join(path,file_name))
            albedo_in = albedo_in[0:768,0:768,:]
            albedo = albedo_in.astype(np.float32)/255
            albedo = np.moveaxis(albedo,-1,0)
        else:
            continue

    # import pdb; pdb.set_trace()
    albedo = torch.from_numpy(albedo).unsqueeze(0)
    albedo = albedo.cuda()

    data = Variable(albedo)
    output = model(data)

    #import pdb; pdb.set_trace()
    output_path = os.path.join(blender_folder,"Texture{}".format(cnt))

    output_normal = output[0,0:3,:,:].cpu().data.numpy()*256
    output_normal = np.moveaxis(output_normal,0,-1)
    normal_file_name = os.path.join(output_path,"{}_normal.png".format(cnt))
    cv2.imwrite(normal_file_name,output_normal)

    output_ao = output[0,3:4,:,:].cpu().data.numpy()*256
    output_ao = np.moveaxis(output_ao,0,-1)
    ao_file_name = os.path.join(output_path,"{}_ao.png".format(cnt))
    cv2.imwrite(ao_file_name,output_ao)

    output_roughness = output[0,2:3,:,:].cpu().data.numpy()*256
    output_roughness = np.moveaxis(output_roughness,0,-1)
    roughness_file_name = os.path.join(output_path,"{}_roughness.png".format(cnt))
    cv2.imwrite(roughness_file_name,output_roughness)

    roughness_file_name = os.path.join(output_path,"{}_roughness.png".format(cnt))

    metallic = np.zeros_like(output_normal)
    metallic_file_name = os.path.join(output_path,"{}_metallic.png".format(cnt))
    cv2.imwrite(metallic_file_name,metallic)

    albedo_file_name = os.path.join(output_path,"{}_basecolor.png".format(cnt))
    cv2.imwrite(albedo_file_name,albedo_in)

    #cv2.imwrite(output_normal, )

    print(roughness_file_name)
    cnt+=1
    if cnt > 11:
        break
