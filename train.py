import torch.utils.data
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
from tqdm import tqdm
from datasets.data_loaders import TimitDataset
from datasets.data_transformations import MFCC
from models.lstm import *

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    dataset = TimitDataset(csv_file='test_data.csv',
                           root_dir='../timit',
                           transform=MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps))

    #Todo used for training the models
num_phonemes = 44

def train(num_epochs=1024, dataset):
    model = LSTM(FLAGS.num_ceps, layers=[], num_phonemes=num_phonemes)
    optimizer = torch.optim.Adam(model.parameters(), lr=)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=3, shuffle=True)
    for epoch in tqdm(range(num_epochs), desc='training epochs'):

        for batch in tqdm(train_loader, desc="training batches"):
            optimizer



if __name__ == '__main__':
    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')

    app.run(main)