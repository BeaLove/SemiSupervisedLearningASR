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

from datasets.data_loaders import TimitDataset
from datasets.data_transformations import MFCC

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    dataset = TimitDataset(csv_file='test_data.csv',
                           root_dir='../timit',
                           transform=MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps))

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
