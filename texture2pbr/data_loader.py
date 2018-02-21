import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from time import time
import cv2
import random
from tqdm import tqdm

class MaterialsDataset(Dataset):
    """ PBR Material dataset
        Set requested_textures to select which textures to return.
        If a material does not have a requested texture it is omitted from dataset.
        Requested textures can be "albedo", "normal", "metallic", "roughness", "ao"
    """
    def __init__(self, root_folder, requested_textures = ["albedo","normal","roughness","ao"], test=False):

        folders = os.listdir(root_folder)
        self.data = dict()
        self.materials_list = list()

        for material in tqdm(folders):
            folder = os.path.join(root_folder,material)
            texture_dict = dict()
            for texture in requested_textures:
                file_name = "{}-{}.png".format(material,texture)

                path = os.path.join(folder,file_name)

                if os.path.exists(path):
                    im = cv2.imread(path)
                    if im is not None:
                        texture_dict[texture] = path

            # check if all requested textures are found
            if len(texture_dict) == len(requested_textures):
                self.data[material] = texture_dict
                self.materials_list.append(material)
        #import pdb; pdb.set_trace()
        # random split
        random.seed(42)
        random.shuffle(self.materials_list)
        num_train = int(len(self.materials_list)*0.75)



        if test:
            self.materials_list = self.materials_list[num_train:]
            print("Test set: {} images".format(len(self.materials_list)))
        else:
            self.materials_list = self.materials_list[:num_train]
            print("Train set: {} images".format(len(self.materials_list)))


    def __len__(self):
        return len(self.materials_list)

    def __getitem__(self, idx):

        material = self.materials_list[idx]



        # get albedo
        albedo_file = self.data[material].get("albedo",None)
        if albedo_file is not None:

            albedo = cv2.imread(albedo_file)
            albedo = np.moveaxis(albedo, -1, 0)
            albedo = albedo.astype(np.float32)/255.0

        # get normal
        normal_file = self.data[material].get("normal",None)
        if normal_file is not None:
            normal = cv2.imread(normal_file)
            normal = np.moveaxis(normal, -1, 0)
            normal = normal.astype(np.float32)/255.0
            #normal /= (np.std(normal)*10)


        # get roughness
        roughness_file = self.data[material].get("roughness",None)
        if roughness_file is not None:
            roughness = cv2.imread(roughness_file)
            roughness = np.moveaxis(roughness, -1, 0)
            roughness = roughness.astype(np.float32)/255.0

        # get roughness
        ao_file = self.data[material].get("ao",None)
        if ao_file is not None:
            ao = cv2.imread(ao_file)
            ao = np.moveaxis(ao, -1, 0)
            ao = ao.astype(np.float32)/255.0


            #import pdb; pdb.set_trace()


            #import pdb; pdb.set_trace()
            # normalize vectors to unit normal
            #normal /= (normal[0,:,:]+normal[0,:,:])
            #normal = normal[:2,:,:] # only RG contains info

        # data augmentation
        #transforms.random_mirror()

        item = {
                "albedo": albedo,
                "roughness": roughness,
                "normal": normal,
                "ao": ao
                }


        return item

def test():
    dataset = MaterialsDataset("PBR_dataset_cleaned")
    import pdb; pdb.set_trace()
    test_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch_idx, batch_item in enumerate(test_loader):
        albedo = batch_item["albedo"]
        normal = batch_item["normal"]
        roughness = batch_item["roughness"]
        ao = batch_item["ao"]
        print(batch_idx)
        #import pdb; pdb.set_trace()
        albedo_grid = make_grid(albedo, nrow=4).numpy()
        albedo_grid = np.moveaxis(albedo_grid,0,-1)
        normal_grid = make_grid(normal, nrow=4).numpy()
        normal_grid = np.moveaxis(normal_grid,0,-1)
        roughness_grid = make_grid(roughness, nrow=4).numpy()
        roughness_grid = np.moveaxis(roughness_grid,0,-1)
        ao_grid = make_grid(ao, nrow=4).numpy()
        ao_grid = np.moveaxis(ao_grid,0,-1)

        plt.subplot(221)
        plt.imshow(cv2.cvtColor(albedo_grid, cv2.COLOR_BGR2RGB))
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(normal_grid, cv2.COLOR_BGR2RGB))
        plt.subplot(223)
        plt.imshow(cv2.cvtColor(roughness_grid, cv2.COLOR_BGR2RGB))
        plt.subplot(224)
        plt.imshow(cv2.cvtColor(ao_grid, cv2.COLOR_BGR2RGB))
        plt.show()

if __name__ == "__main__":
    test()
