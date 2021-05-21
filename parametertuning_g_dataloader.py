from absl import app
from absl import flags
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from absl import app
from scipy.io import wavfile
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter



import pandas as pd
import os
import scipy.io
import subprocess
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch
import test
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import pickle
from functools import partial
from datasets.data_loaders_2 import TimitDataset
from datasets.data_transformations import MFCC
from datasets.data_transformations import Phonemes
from datasets.corpus import *
from models.lstm1 import LSTM
from torch.optim.lr_scheduler import CosineAnnealingLR


logger = SummaryWriter()

os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"


def main(argv):
    """ This is our main method.
    """
    del argv  # Unused.

    data_dir = 'timit'
    gpus_per_trial = 0.5
    max_num_epochs = 100
    num_samples = 8
    config = {
        "num layers" : tune.sample_from(lambda _: np.random.randint(1, 5)), #number of layers in the model
        "units" : tune.choice([25, 50, 75, 100, 150]), #number of hidden nodes in each layer
        "dropout": tune.sample_from(lambda _: np.random.random()*0.5),
        "weight decay": tune.loguniform(1e-5, 1e-2),
        'n_fft': 512,
        'num_ceps': 13,
        'preemph': 0.97,

    }
    # ...

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        training_run,
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True)

    best_trial = result.get_best_trial("loss", "min", "last")
    with open('best_trail_results.txt', 'a') as file:
        print("Best trial config: {}".format(best_trial.config))
        file.write("Best trial config: {} \n".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        file.write("Best trial final validation loss: {}\n".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))
        file.write("Best trial final validation accuracy: {} \n".format(
            best_trial.last_result["accuracy"]))


# ------------------------------------------- hyperparameter setup------------------------------
def training_run(config):
    # Initialize a Corpus object
    example_file_dir = "/data/TRAIN/DR1/FCJF0/SA1"  # SA1.wav.WAV
    # dataset_dir = "/home/georgmosh/Documents/SpeechLabs/dt2119_semisup_project/SemiSupervisedLearningASR-main/timit"
    dataset_dir = 'timit'
    corpus = Corpus(dataset_dir, example_file_dir)  # TIMIT corpus
    phonemes = corpus.get_phonemes()  # List of phonemes
    targets = len(phonemes)  # Number of categories

    # Load the TIMIT dataset
    train_dataset = TimitDataset(csv_file='train_data.csv',
                                 root_dir=dataset_dir,
                                 corpus=corpus,
                                 transform=MFCC(n_fft=config['n_fft'],
                                                preemphasis_coefficient=config['preemph'],
                                                num_ceps=config['num_ceps']),
                                 transcription=Phonemes(n_fft=config['n_fft'],
                                                        preemphasis_coefficient=config['preemph'],
                                                        num_ceps=config['num_ceps'],
                                                        corpus=corpus))
    print(len(train_dataset))
    test_dataset = TimitDataset(csv_file='test_data.csv',
                                root_dir=dataset_dir,
                                corpus=corpus,
                                transform=MFCC(n_fft=config['n_fft'],
                                               preemphasis_coefficient=config['preemph'],
                                               num_ceps=config['num_ceps']),
                                transcription=Phonemes(n_fft=config['n_fft'],
                                                       preemphasis_coefficient=config['preemph'],
                                                       num_ceps=config['num_ceps'],
                                                       corpus=corpus))

    # Get the MFCC coefficients
    train_data, max_len = getMFCCFeatures(train_dataset, oneTensor=False)
    test_data, max_len_test = getMFCCFeatures(test_dataset, oneTensor=False)

    # Get the phonemes per frame (as percentages)
    train_targets = getTargetPhonemes(train_dataset, max_len, corpus, oneTensor=False, mode="indices")
    test_targets = getTargetPhonemes(test_dataset, max_len, corpus, oneTensor=False, mode="indices")

    # Create directories

    # Configure model
    save_path = "../trained_models/raytuneTrial.pth"

    model = LSTM(config['num_ceps'], corpus.get_phones_len(), units_per_layer=config['units'],
                 num_layers=config['num layers'], dropout=config['dropout'])

    # Configuring the Optimizer (ADAptive Moments)

    # Train model
    trainModel(train_data, train_targets, 4620, model, weight_decay_config=config['weight decay'])



# ------------------------------------------- ON DIRECTORY -------------------------------------------

def makedirs(path):
    Path(path).mkdir(parents=False, exist_ok=True)


# ------------------------------------------- LOSS PLOTTING -------------------------------------------

def plot_loss(loss_train, loss_val, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Average training vs validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Write loss plot to disk
    #plt.savefig(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'loss_plot.png')))


# ------------------------------------------- LSTM TRAINING -------------------------------------------

def trainModel(train_data, train_targets, num_data, model, weight_decay_config, val_split = 0.15):

    val_split = num_data - int(num_data * val_split)

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')
    logger = SummaryWriter()

    # Congifuring the model
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=weight_decay_config,
                                 amsgrad=False)
    model.to(device)
    scheduler = CosineAnnealingLR(tunT_max=462000) #max no iterations eg num_v=batches * max_epochs
    for i in range(0, val_split):
        sample = train_data[i].type(torch.FloatTensor)
        target = train_targets[i].type(torch.LongTensor)
        sample = torch.reshape(sample, (sample.shape[0], 1, sample.shape[1]))
        sample = sample.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        prediction = model.forward(sample)
        loss_val = loss(torch.squeeze(prediction, dim=1), target)
        loss_val.backward()
        optimizer.step()
        scheduler.step()
        logger.add_scalar('train loss', loss_val)
        logger.add_scalar('lr', scheduler.get_lr())
        tune.report(loss_val)
        tune.report(scheduler.get_lr())

    model.eval()
    for i in range(val_split, num_data):
        sample = train_data[i].type(torch.FloatTensor)
        target = train_targets[i].type(torch.LongTensor)
        sample = torch.reshape(sample, (sample.shape[0], 1, sample.shape[1]))
        sample = sample.to(device)
        target = target.to(device)
        output = model.forward(sample)
        val_loss = loss(output.squeeze(), target.squeeze())
        logger.add_scalar('val loss', val_loss)
        tune.report(val_loss)

    model.train()



def testModel(test_data, test_targets, num_data, model):
    correct = 0
    total = 0

    model.eval()

    bar = tqdm(range(num_data))

    for i in bar:
        sample = test_data[i]
        target = test_targets[i]
        sample = torch.nn.utils.rnn.pad_packed_sequence(sample)
        output = model.forward(sample)
        prediction = torch.max(output, dim=0)
        correct += (prediction == target).float().sum()
        total += target.shape[0]
    accuracy = correct / total * 100

    return accuracy


# ------------------------------------------- DATA LOADING -------------------------------------------

def getMFCCFeatures(dataset, zeropad=False, oneTensor=False):
    """ This method computes the MFCC coefficients per frame.
         When frames are less than the maximum amount does zero-padding.
         @returns tensors of MFCC coefficients of the same length
    """
    features = []
    tensors = []
    max_frames = -1

    for i in range(len(dataset)):
        sample = dataset[i]
        audio = np.asarray(sample['audio'])
        if (max_frames < audio.shape[0]):
            max_frames = audio.shape[0]
        if (zeropad == True):
            features.append(audio)
        else:
            tensors.append(torch.tensor(audio.tolist(), dtype=torch.float))

    if (zeropad == True):
        # zero-padding for equal length
        for i in range(len(dataset)):
            audio_new = np.zeros((max_frames, features[i].shape[1]))
            audio_new[0:features[i].shape[0], :] = features[i]
            tensors.append(torch.tensor(audio_new.tolist(), dtype=torch.float))

    if (oneTensor == True):
        whole = tensors[0].numpy()
        for i in range(1, len(dataset)):
            whole = np.concatenate((whole, tensors[i].numpy()), axis=0)
        tensors = torch.tensor(whole.tolist(), dtype=torch.float)

    return tensors, max_frames


def getTargetPhonemes(dataset, max_frames, corpus, zeropad=False, oneTensor=False, mode="indices"):
    """ This method computes the target phonemes as percentages per frame.
         @returns tensors of phonemes per frame
    """
    tensors = []
    targets = []

    for i in range(len(dataset)):
        sample = dataset[i]
        phoneme_list = sample['phonemes_per_frame']
        sample_targets = []

        for j in range(len(phoneme_list)):
            if (mode == "indices"):
                # using only one phoneme explicitly imposed --> first phoneme index
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_id = corpus.get_phones_ID(the_one_phoneme)
                sample_targets.append(the_one_phoneme_id)
            elif (mode == "percentages"):
                if (len(phoneme_list[j]) == 1):
                    # only one phoneme in this frame --> one hot encoding
                    the_one_phoneme = phoneme_list[j][0][2]
                    the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[0]
                    sample_targets.append(the_one_phoneme_one_hot)
                else:
                    # more than one phonemes in this frame --> probabilities
                    complex_one_hot = np.zeros(corpus.get_phones_len()).tolist()
                    for k in range(len(phoneme_list[j])):
                        the_kth_phoneme = phoneme_list[j][k][2]
                        the_kth_phoneme_ID = corpus.get_phones_ID(the_kth_phoneme)
                        the_kth_phoneme_percentage = phoneme_list[j][k][3]
                        complex_one_hot[the_kth_phoneme_ID] = the_kth_phoneme_percentage
                    sample_targets.append(complex_one_hot)
            elif (mode == "onehot"):
                # using only one phoneme explicitly imposed --> one hot encoding
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_id = corpus.get_phones_ID(the_one_phoneme)
                the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[0]
                sample_targets.append(the_one_phoneme_one_hot)
            else:
                print("Wrong mode!")
                break

        if (zeropad == True):
            for j in range(len(phoneme_list), max_frames):
                # adding a silence target for the extra frames (zero-padded additions)
                silence = corpus.get_silence()
                if (mode == "percentages" or mode == "onehot"):
                    silence_one_hot = corpus.phones_to_onehot([silence])[0]
                    sample_targets.append(silence_one_hot)
                elif (mode == "indices"):
                    silence_id = corpus.get_phones_ID(silence)
                    sample_targets.append(silence_id)

        tensors.append(torch.tensor(sample_targets, dtype=torch.long))

    if (oneTensor == True):
        whole = tensors[0].numpy()
        for i in range(1, len(dataset)):
            whole = np.concatenate((whole, tensors[i].numpy()), axis=0)
        tensors = torch.tensor(whole.tolist(), dtype=torch.long)

    return tensors


if __name__ == '__main__':


    app.run(main)
