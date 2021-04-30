from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio

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
        #TODO: break up wav file according to phonemes in phn file
            '''then do preprocessing to produce a 3D tensor of [phoneme, frames, mfcc]'''
        phonemes_name = path + '.PHN'

        with open(phonemes_name, 'r') as phonemes_file:
            phonemes = [l.strip().split(' ') for l in phonemes_file]
            phonemes = [[int(hd), int(tl), w] for (hd, tl, w) in phonemes]

        phonemes = list(phonemes)
        #TODO return all phonemes in each sample as tensor
        '''with each row corresponding to the correct phoneme, one-hot encoded
        '''

        sample = {'audio': audio, 'phonemes':  phonemes}

        if self.transform:
            sample = self.transform(sample, sample_rate)

        return sample
