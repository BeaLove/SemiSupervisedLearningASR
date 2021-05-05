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

    def __init__(self, csv_file, root_dir, pre_epmh, num_ceps, n_fft, frame_size, frame_shift):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.audio_frame = self.audio_frame[self.audio_frame['is_converted_audio'] == True]

        self.root_dir = root_dir
        #self.transform = transform

        self.use_cache = False
        self.cache = {}
        self.pre_emph = pre_epmh
        self.num_ceps = num_ceps
        self.n_fft = n_fft
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.phones = ["b", "d", "g", "p", "t", "k", "dx", "q",
                  "jh", "ch", "s", "sh", "z", "zh", "f", "th",
                  "v", "dh", "m", "n", "ng", "em", "en", "eng",
                  "nx", "l", "r", "w", "y", "hh", "hv", "el",
                  "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay",
                  "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er",
                  "ax", "ix", "axr", "ax-h", "pau", "epi",
                  # closure portion of stops
                  "bcl", "dcl", "gcl", "pcl", "tcl", "kcl",
                  # silence
                  "h#"]
        self.fullPhones2int = dict([(key,val) for val,key in enumerate(self.phones)])
        self.mergedPhones2int = {'iy': 0, 'ih': 1, 'eh': 2, 'ae': 3, 'ix': 4, 'ax': 5, 'ah': 6, 'uw': 7, 'ux': 7,
                                 'uh': 8, 'ao': 9, 'aa': 10, 'ey': 11, 'ay': 12, 'oy': 13, 'aw': 14, 'ow': 15,
                                 'l': 16, 'el': 17, 'r': 18, 'y': 19, 'w': 20, 'er': 21, 'axr': 21,
                                 'm': 22, 'em': 22, 'n': 23, 'nx': 23, 'en': 24, 'ng': 25, 'eng': 25,
                                 'ch': 26, 'jh': 27, 'dh': 28, 'b': 29, 'd': 30, 'dx': 31, 'g': 32,
                                 'p': 33, 't': 34, 'k': 35, 'z': 36, 'zh': 37, 'v': 38, 'f': 39,
                                 'th': 40, 's': 41, 'sh': 42, 'hh': 43, 'hv': 43,
                                 'pcl': 44, 'tcl': 44, 'kcl': 44, 'qcl': 44, 'bcl': 45,
                                 'dcl': 45, 'gcl': 45, 'epi': 46, 'sil': 47, 'h#': 47, '#h': 47, 'pau': 47, 'q': 48, 'ax-h': 49}
        self.num_labels = max(self.mergedPhones2int.values()) + 1
        self.max_length = 777

    def __len__(self):
        return len(self.audio_frame)

    def __getitem__(self, idx):
        '''args: idx of data sample
            operation: extracts wav audio and phn file (phone transcription) of sample, enframes
            audio and returns with 13 MFCC coefficients, computes correct phoneme for each frame
            and one-hot encodes a target num_frames x num_phones
            out: sample, target'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.root_dir, 'data',
                            self.audio_frame.iloc[idx]['path_from_data_dir'])
        #print(path)
        path = os.path.splitext(path)[0]
        path = os.path.splitext(path)[0]



        audio_name = path + '.WAV.wav'
        #print("audio", audio_name)

        audio, self.sample_rate = torchaudio.load(audio_name)
        phonemes_name = path + '.PHN'
        #print('phonemes: ', phonemes_name)
        with open(phonemes_name, 'r') as phonemes_file:
            start = []
            stop = []
            phonemes = []
            chunks = []
            number_of_frames = []
            frame_labels = []
            lines = phonemes_file.readlines()
            last = lines[-1]
            for l in lines:
                phoneme = l.strip().split()
                start.append(int(phoneme[0]))
                stop.append(int(phoneme[1]))
                if l == last:
                    phoneme[1] = audio.shape[1]
                chunk = int(phoneme[1]) - int(phoneme[0])
                chunks.append(int(phoneme[1]) - int(phoneme[0]))
                num_frames = (chunk/self.sample_rate) / 0.01
                number_of_frames.append(num_frames)
                if num_frames < 1:
                    frame_labels.append(phoneme[2])
                else:
                    frame_labels += [phoneme[2]]*round(num_frames)
                phonemes.append(phoneme[2])

        '''for i in range(len(phonemes)):
            print("start {} stop {} phoneme {} number of frames {}".format(start[i], stop[i], phonemes[i], number_of_frames[i]))
        '''

        sample = self.get_MFCC(audio)
        pad = self.max_length - sample.shape[0]
        #torch.nn.functional.pad(sample, (0, 0, 0, 10))


        frame_labels = frame_labels[:sample.shape[0]]

        encoded_phones = []
        #for phone in frame_labels:
         #   encoded_phones.append(self.getPhoneCode(phone))
        targets = torch.tensor([self.oneHot(self.num_labels, self.getPhoneCode(phone)) for phone in frame_labels] , dtype=torch.float64)
        #targets = torch.nn.utils.rnn.pad_packed_sequence(targets, total_length=777)
        #print(targets.shape)
        #sample = {'audio': sample, 'targets': targets}
        return sample, targets



    def getPhoneCode(self, phone):
        return self.mergedPhones2int[phone]

    def oneHot(self, size, idx):
        return np.eye(size)[idx]

    def get_MFCC(self, sample):

        #frame_length = self.n_fft / self.sample_rate * 1000.0
        #frame_shift = frame_length / 2.0

        params = {
            "channel": 0,
            "dither": 0.0,
            "window_type": "hamming",
            "frame_length": self.frame_size,
            "frame_shift": int(self.frame_size/2),
            "remove_dc_offset": False,
            "round_to_power_of_two": False,
            "sample_frequency": self.sample_rate,
            "preemphasis_coefficient": self.pre_emph,
            "num_ceps": self.num_ceps
        }

        #audio = torch.tensor(audio, dtype=torch.float)
        frames_mfcc = torchaudio.compliance.kaldi.mfcc(sample, **params)
        return frames_mfcc

csv_filename = 'train_data.csv'
root_dir = '../timit'

train_set = TimitDataset(csv_file=csv_filename, root_dir=root_dir, pre_epmh=0.97, num_ceps=13, n_fft=512, frame_size=20, frame_shift=10)
print('dataset size', len(train_set))
max_len = 0
for i in range(len(train_set)):
    sample, target = train_set.__getitem__(i)
    print(sample.shape, target.shape)
    if sample.shape[0] > max_len:
        max_len = sample.shape[0]
        print("new max: ", max_len)
print("max", max_len)
print("dataset size", len(train_set))
test_file = 'test_data.csv'

test_set = TimitDataset(csv_file=test_file, root_dir=root_dir, pre_epmh=0.97, num_ceps=13, n_fft=512, frame_size=20, frame_shift=10)

max_len = 0
for i in range(len(test_set)):
    sample, target = test_set.__getitem__(i)
    print(sample.shape, target.shape)
    if sample.shape[0] > max_len:
        max_len = sample.shape[0]
        print("new max: ", max_len)
print("max", max_len)
print("dataset size", len(test_set))

'''sample = dataset.__getitem__(0)
print(sample)'''