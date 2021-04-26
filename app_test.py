from absl import app
from absl import flags
from pathlib import Path
import pandas as pd
import os
from scipy.io import wavfile
import scipy.io
import subprocess
import torchaudio

from datasets.data_loaders import TimitDataset
from datasets.data_transformations import MFCC

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    print("Hello these are your settings: ")
    print(FLAGS.winlen)
    print(FLAGS.winstep)

    dataset = TimitDataset(csv_file='test_data.csv',
                           root_dir='../timit',
                           transform=MFCC())

    for i in range(len(dataset)):
        sample = dataset[i]

        print(sample)

        if i == 3:
            break


if __name__ == '__main__':
    flags.DEFINE_float('winlen', 0.02, 'windowing length')
    flags.DEFINE_float('winstep', 0.01, 'Windowing step')
    flags.DEFINE_integer('numcep', 13, 'Number of MFCC coefficients')
    flags.DEFINE_integer('nfilt', 26, 'Number of filters')
    flags.DEFINE_float('preemph', 0.97, 'Preemphasis filter coefficient')
    app.run(main)
