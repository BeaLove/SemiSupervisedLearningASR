from absl import app
from absl import flags
from pathlib import Path
import pandas as pd
import os
from scipy.io import wavfile
import scipy.io
import subprocess
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.data_loaders import TimitDataset
from datasets.data_transformations import MFCC
from datasets.corpus import *

FLAGS = flags.FLAGS

def main(argv):
    """ This is our main method.
    """
    del argv  # Unused.
    
    # Initialize a Corpus object
    example_file_dir = "/data/TRAIN/DR1/FCJF0/SA1"  #SA1.wav.WAV
    dataset_dir = "/home/georgmosh/Documents/SpeechLabs/dt2119_semisup_project/SemiSupervisedLearningASR-main/timit"
    # dataset_dir = '../timit'
    corpus = Corpus(dataset_dir, example_file_dir) # TIMIT corpus
    phonemes = corpus.get_phonemes()  # List of phonemes
    targets = len(phonemes)  # Number of categories

    # Load the TIMIT dataset
    dataset = TimitDataset(csv_file='test_data.csv',
                           root_dir = dataset_dir,
                           transform=MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps))

    # Get the MFCC coefficients
    train_data = getMFCCFeatures(dataset)

    # Toy example: loading phonemes for one audio file
    phones_start_SA1, phones_stop_SA1, phonemes_SA1 = corpus.get_phone_transcription(example_file_dir)  # SA1.wav.WAV
    phonemes_ids_SA1 = []
    phonemes_ids_SA1_1hot = []
    
    # Indices of the phonemes
    for phoneme in phonemes_SA1:
        phonemes_ids_SA1.append(corpus.get_phones_ID(phoneme))
        
    # One-hot representation of the indices of the phonemes
    phonemes_ids_SA1_1hot = corpus.phones_to_onehot(phonemes_SA1)
    
def getMFCCFeatures(dataset):
    """ This method computes the MFCC coefficients per frame.
         When frames are less than the maximum amount does zero-padding.
         @returns arrays of MFCC coefficients of the same length
    """
    features = []
    tensors = []
    max_frames = -1

    for i in range(len(dataset)):
            sample = dataset[i]
            audio = np.asarray(sample['audio'])
            if(max_frames < audio.shape[0]):
                max_frames = audio.shape[0]
            features.append(audio)
    
    for i in range(len(dataset)):
            audio_new = np.zeros((max_frames, features[i].shape[1]))
            audio_new[0:features[i].shape[0],:] = features[i]
            tensors.append(torch.tensor(audio_new.tolist(), dtype=torch.long))
    
    return tensors
    
    
def test1(dataset):
    for i in range(len(dataset)):
            sample = dataset[i]

            audio = np.asarray(sample['audio'])

            print("Audio shape: ", audio.shape)

            plt.pcolormesh(audio)
            plt.show()

            if i == 1:
                break

if __name__ == '__main__':
    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')

    app.run(main)
