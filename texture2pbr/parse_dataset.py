import os
import shutil
import cv2

"""
    This cleaning script iterates through folders and finds textures of different types
    Textures are then renamed with prefix (albedo, normal, metallic, roughness, ao)
"""

INPUT_FOLDER = "/Users/sundholm/Data/PBR_dataset"
OUTPUT_FOLDER = "/Users/sundholm/Data/PBR_dataset_cleaned"

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
        # find normal map
        if string_contains(file_name, ["normal"]):
            texture_dict[folder]["normal"] = os.path.join(INPUT_FOLDER,folder,file_name)
        # find ambien occlusion
        if string_contains(file_name, ["ao", "ambient", "occ"]):
            texture_dict[folder]["ao"] = os.path.join(INPUT_FOLDER,folder,file_name)

# create
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for material in texture_dict:
    folder = os.path.join(OUTPUT_FOLDER,material)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for texture_type in texture_dict[material]:
        src = texture_dict[material][texture_type]
        dst = os.path.join(folder,"{}-{}.png".format(material,texture_type))
        print(dst)
        #try:
            #shutil.copyfile(src,dst)
        im = cv2.imread(src)
        im = cv2.resize(im,(128,128))
        cv2.imwrite(dst,im)
        #except:
        #import pdb; pdb.set_trace()
