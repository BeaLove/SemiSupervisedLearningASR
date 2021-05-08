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
import torch.utils.data.dataloader
import tqdm
import datetime

from datasets.TIMITdataset import TimitDataset
#from datasets.data_transformations import MFCC

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    model_name = 'vanillaLSTMfullylabeled'
    root_dir = os.path.abspath('timit')
    test_data = TimitDataset(csv_file='test_data.csv', root_dir=root_dir,
                             pre_epmh=FLAGS.preemphasis_coefficient,
                             num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                             frame_shift=FLAGS.frame_shift)

    accuracy = test(model_name, test_data)
    date = datetime.date()
    with open('vanillaLSTMfullylabeledtest.txt', 'a') as file:
        file.write('Model: {} date: {} accuracy: {}'.format(model_name, date, accuracy))
    print("model accuracy: ", model_name, accuracy)



    #Todo used for testing locally on CPU

def test(model_name, test_data):
    try:
        model_path = os.path.abspath(model_name)
    except:
        print('trained model not found')
        raise ModuleNotFoundError

    model = torch.load(model_path)
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=3)
    correct = 0
    total = 0
    with torch.no_grad():
        for point in tqdm(test_loader):
            sample, target = point
            output = model.forward(sample)
            prediction = torch.max(output, dim=0)
            correct += (prediction == target).float().sum()
            total += target.shape[0]
    accuracy = correct/total * 100

    return accuracy

if __name__ == '__main__':
    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer('num_ceps', 13, ' Number of cepstra in MFCC computation')
    flags.DEFINE_integer('frame_len', 20, 'Frame length in ms')
    flags.DEFINE_integer('frame_shift', 10, 'frame shift in ms')
    flags.DEFINE_string('csv_file', 'train_data.csv', 'csv file with TIMIT data info')
    flags.DEFINE_string('root_dir', '../timit', 'root directory for TIMIT data')
    flags.DEFINE_integer('epochs', 1024, 'number of epochs')
    flags.DEFINE_float('lr', 0.001, 'learning rate')

    app.run(main)
