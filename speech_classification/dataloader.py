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
        self.num_classes = len(self.class_list)

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
        fs, wave_sequence = wav.read(path)

        # make sure all sequences are same size
        pad_size = max(16000-len(wave_sequence),0)
        wave_sequence = np.concatenate([wave_sequence,np.zeros(pad_size,dtype=wave_sequence.dtype)])
        wave_sequence = wave_sequence[:16000]

        f, t, Z = signal.stft(wave_sequence, fs, nperseg=128)
        Z = np.dstack((Z.real,Z.imag))
        Z = np.swapaxes(Z,0,2)
        Z = np.swapaxes(Z,1,2)
        widths = np.arange(1, 2048)

        class_name = path.split("/")[-2]
        print(len(wave_sequence))

        item = {
            "class_idx": self.class_lookup[class_name],
            "class_name": class_name,
            "wave_sequence": np.expand_dims(wave_sequence,0).astype(np.float32)/np.std(wave_sequence),
            "fs": fs
            #"spectrum": Z,
        }
        return item

def test():
    dataset = AudioDataset("/Users/sundholm/Data/kaggle_tf_sound/train/audio")

    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for batch_idx, batch_item in enumerate(test_loader):
        Z = batch_item["spectrum"]

        #plt.imshow(np.abs(Z[0,:,:],Z[1,:,:]))
        #import pdb; pdb.set_trace()

        #Z = np.abs(Z[0,0,:,:].numpy(),Z[0,1,:,:].numpy())
        #Z = Z[0,:,:,:].numpy()
        import pdb; pdb.set_trace()
        #plt.imshow(np.abs(Z[0,:,:], Z[1,:,:])*np.angle(Z[0,:,:], Z[1,:,:]))
        #import pdb; pdb.set_trace()
        #plt.imshow(normal_grid)
        plt.show()

if __name__ == "__main__":
    test()
