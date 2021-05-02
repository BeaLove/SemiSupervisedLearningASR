from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
import data_transformations
import ctypes
import multiprocessing as mp

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TimitDataset(Dataset):
    """Timit dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.audio_frame = self.audio_frame[self.audio_frame['is_phonetic_file'] == True]

        self.root_dir = root_dir
        self.transform = transform

        self.use_cache = False
        self.cache = {}

    def __len__(self):
        return len(self.audio_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.root_dir, 'data',
                            self.audio_frame.iloc[idx]['path_from_data_dir'])
        path = os.path.splitext(path)[0]

        audio_name = path + '.WAV.wav'

        audio, sample_rate = torchaudio.load(audio_name)
        phonemes_name = path + '.PHN'

        with open(phonemes_name, 'r') as phonemes_file:
            start = []
            stop = []
            phonemes = []
            chunks = []
            for l in phonemes_file:
                phoneme = l.strip().split()
                start.append(int(phoneme[0]))
                stop.append(int(phoneme[1]))
                chunks.append(int(phoneme[1]) - (phoneme[0]))
                phonemes.append(phoneme[2])

        for i in range(len(phonemes)):
            print("start {} stop {} phoneme {}".format(start[i], stop[i], phonemes[i]))
        #TODO return all phonemes in each sample as tensor
        '''with each row corresponding to the correct phoneme, one-hot encoded
        '''
        split_audio = torch.split(audio, split_size_or_sections=chunks)
        #TODO: enframe splits, encode target phonemes
        sample = {'audio': audio, 'phonemes':  phonemes}

        if self.transform:
            sample = self.transform(sample, sample_rate)

        return sample

csv_filename = 'train_data.csv'
root_dir = '../timit'

dataset = TimitDataset(csv_file=csv_filename, root_dir=root_dir)
sample = dataset.__getitem__(0)
print(sample)