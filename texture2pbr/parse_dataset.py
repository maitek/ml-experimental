import os
import shutil
import cv2
from torchvision.utils import make_grid
import numpy as np
from concurrent import futures

"""
    This cleaning script iterates through folders and finds textures of different types
    Textures are then renamed with prefix (albedo, normal, metallic, roughness, ao)
"""

INPUT_FOLDER = "PBR_dataset"
OUTPUT_FOLDER = "PBR_dataset_256"


folders = [x for x in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER,x))]

texture_dict = dict()

def string_contains(string, sub_strings):
    for sub in sub_strings:
        if sub in string.lower():
            return True
    else:
        return False

for folder in folders:
    files = os.listdir(os.path.join(INPUT_FOLDER,folder))
    files = [x for x in files if x.endswith(".png")]

    texture_dict[folder] = dict()

    for file_name in files:
        file_name_replace = file_name.replace('_', '-')
        ending = file_name_replace.split("-")[-1]

        # find albedo
        if string_contains(file_name, ["color", "albedo", "alb"]):
            texture_dict[folder]["albedo"] = os.path.join(INPUT_FOLDER,folder,file_name)
        # find metallic
        if string_contains(file_name, ["metal"]):
            texture_dict[folder]["metallic"] = os.path.join(INPUT_FOLDER,folder,file_name)
        # find roughness
        if string_contains(file_name, ["rough"]):
            texture_dict[folder]["roughness"] = os.path.join(INPUT_FOLDER,folder,file_name)
        # find normal mapfrom concurrent import futures
        if string_contains(file_name, ["normal"]):
            texture_dict[folder]["normal"] = os.path.join(INPUT_FOLDER,folder,file_name)
        # find ambien occlusion
        if string_contains(file_name, ["ao", "ambient", "occ"]):
            texture_dict[folder]["ao"] = os.path.join(INPUT_FOLDER,folder,file_name)
        # find ambien occlusion
        if string_contains(file_name, ["height",]):
            texture_dict[folder]["height"] = os.path.join(INPUT_FOLDER,folder,file_name)

# create
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def save_material(material):
    folder = os.path.join(OUTPUT_FOLDER,material)
    crop_size = 256
    num_crops = 10
    for crop_idx in range(num_crops):

        for idx, texture_type in enumerate(texture_dict[material]):
            src = texture_dict[material][texture_type]

            out_folder = "{}_{}".format(folder,crop_idx)
            material_name = "{}_{}".format(material,crop_idx)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            dst = os.path.join(out_folder,"{}-{}.png".format(material_name,texture_type))
            print(dst)
            #try:
                #shutil.copyfile(src,dst)
            im = cv2.imread(src)
            if idx == 0:
                h, w, d = im.shape
                rx = np.random.randint(0,w-crop_size)
                ry = np.random.randint(0,h-crop_size)

            crop = im[rx:rx+crop_size,rx:rx+crop_size]
            #import pdb; pdb.set_trace()
            #im = cv2.resize(im,(256,256))
            cv2.imwrite(dst,crop)
            #except:
            #import pdb; pdb.set_trace()




#for material in texture_dict:
#    save_material(material)

# parallel processing
with futures.ProcessPoolExecutor() as executor:
        executor.map(save_material, texture_dict)
