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
    print("Hello these are your settings: ")
    print("Window length flag:", FLAGS.winlen)
    print("Window step flag: ", FLAGS.winstep)

    dataset = TimitDataset(csv_file='test_data.csv',
                           root_dir='../timit',
                           transform=MFCC(n_mfcc=FLAGS.n_mfcc, preemph=FLAGS.preemph))

    for i in range(len(dataset)):
        sample = dataset[i]

        audio = np.asarray(sample['audio'])[0]

        print("Audio shape: ", audio.shape)

        plt.pcolormesh(audio.T)
        plt.show()

        if i == 1:
            break


if __name__ == '__main__':
    flags.DEFINE_float('winlen', 0.02, 'windowing length')
    flags.DEFINE_float('winstep', 0.01, 'Windowing step')
    flags.DEFINE_integer('n_mfcc', 13, 'Number of MFCC coefficients')
    flags.DEFINE_integer('nfilt', 26, 'Number of filters')
    flags.DEFINE_float('preemph', 0.97, 'Preemphasis filter coefficient')
    app.run(main)
