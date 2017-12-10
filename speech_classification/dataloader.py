import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from time import time
import random
import scipy.io.wavfile as wav
from scipy import signal

class AudioDataset(Dataset):
    def __init__(self, root_folder, train=True):

        self.class_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        self.class_lookup = dict()
        for idx, item in enumerate(self.class_list):
            self.class_lookup[item] = idx


        folders = os.listdir(root_folder)

        folders = [x for x in folders if x in self.class_list]

        self.users = set()
        self.train = train
        # split data by user
        for class_folder in folders:
            audio_files = os.listdir(os.path.join(root_folder,class_folder))
            for audio_file in audio_files:
                self.users.add(audio_file.split("_")[0])


        sorted_users = sorted(list(self.users))
        self.train_users = sorted_users[:int(len(sorted_users)*0.8)]
        self.val_users = sorted_users[int(len(sorted_users)*0.8):]
        self.train_data = list()
        self.val_data = list()

        for idx, class_folder in enumerate(folders):
            audio_files = os.listdir(os.path.join(root_folder,class_folder))
            for jdx, audio_file in enumerate(audio_files):
                user_id =  audio_file.split("_")[0]
                audio_path = os.path.join(root_folder,class_folder,audio_file)
                if user_id in self.train_users:
                    self.train_data.append(audio_path)
                else:
                    self.val_data.append(audio_path)

        print("== Train ==")
        #for class_name in self.train_data.keys():
        #   print(class_name,len(self.train_data[class_folder]))

        print("== Validation ==")
        #for class_name in self.val_data.keys():
        #    print(class_name,len(self.val_data[class_folder]))

    def __len__(self):
        if self.train:
            length = len(self.train_data)
        else:
            length = len(self.val_data)
        return length

    def __getitem__(self, idx):

        if self.train:
            path = self.train_data[idx]
        else:
            path = self.val_data[idx]
        fs, wave_file = wav.read(path)
        f, t, Z = signal.stft(wave_file, fs, nperseg=128)

        class_name = path.split("/")[-2]

        item = {
            "y": self.class_lookup[class_name],
            "class_name": class_name,
            "wave_file": wave_file,
            "fs": fs,
            "spectrum": np.abs(Z),
        }
        return item

def test():
    dataset = AudioDataset("/Users/sundholm/Data/kaggle_tf_sound/train/audio")

    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for batch_idx, batch_item in enumerate(test_loader):
        Z = batch_item["spectrum"]
        import pdb; pdb.set_trace()
        plt.imshow(Z.numpy()[0,:,:])
        #import pdb; pdb.set_trace()
        #plt.imshow(normal_grid)
        plt.show()

if __name__ == "__main__":
    test()
