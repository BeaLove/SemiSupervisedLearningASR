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

from datasets.data_loaders_2 import TimitDataset
from datasets.data_transformations import MFCC
from datasets.data_transformations import Phonemes
from datasets.corpus import *
from models.lstm1 import LSTM

FLAGS = flags.FLAGS

def main(argv):
    """ This is our main method.
    """
    del argv  # Unused.
    
    set_seeds(0)
    
    # Initialize a Corpus object
    example_file_dir = "/data/TRAIN/DR1/FCJF0/SA1"  #SA1.wav.WAV
    #dataset_dir = "/home/georgmosh/Documents/SpeechLabs/dt2119_semisup_project/SemiSupervisedLearningASR-main/timit"
    dataset_dir = 'timit'
    corpus = Corpus(dataset_dir, example_file_dir) # TIMIT corpus
    phonemes = corpus.get_phonemes()  # List of phonemes
    targets = len(phonemes)  # Number of categories

    # Load the TIMIT dataset
    train_dataset = TimitDataset(csv_file = 'train_data.csv',
                           root_dir = dataset_dir,
                           corpus = corpus,
                           transform = MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps),
                           transcription = Phonemes(n_fft=FLAGS.n_fft,
                                           preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                           num_ceps=FLAGS.num_ceps,
                                           corpus = corpus))
    
    test_dataset = TimitDataset(csv_file = 'test_data.csv',
                           root_dir = dataset_dir,
                           corpus = corpus,
                           transform = MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps),
                           transcription = Phonemes(n_fft=FLAGS.n_fft,
                                           preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                           num_ceps=FLAGS.num_ceps,
                                           corpus = corpus))

    # Get the MFCC coefficients
    train_data, max_len = getMFCCFeatures(train_dataset, oneTensor = False)
    test_data, max_len_test = getMFCCFeatures(test_dataset, oneTensor = False)
    
    # Get the phonemes per frame (as percentages)
    train_targets = getTargetPhonemes(train_dataset, max_len, corpus, oneTensor = False, mode = "indices")
    test_targets = getTargetPhonemes(test_dataset, max_len, corpus, oneTensor = False, mode = "indices")
    
    # Create directories
    makedirs(FLAGS.results_save_dir)
    makedirs(os.path.join(FLAGS.results_save_dir, 'checkpoints'))

    # Configure model
    model_name = FLAGS.name
    save_folder = os.path.abspath(FLAGS.results_save_dir)
    save_path = os.path.join(save_folder, model_name)
    
    # Train model
    epochNum = FLAGS.num_epochs
    model, avg_val_losses, avg_train_losses, train_accuracies, val_accuracies, test_accuracies = trainModel(train_data, train_targets, test_data, test_targets, len(train_dataset), len(test_dataset), corpus, num_epochs=epochNum)
    torch.save(model.state_dict(), save_path)
    timestamp = str(datetime.now())

    acc = testModel(test_data, test_targets, len(test_dataset), model)
    plot_loss(avg_train_losses, avg_val_losses, epochNum)
    plot_accuracy(train_accuracies, val_accuracies, test_accuracies, epochNum)
    
    # Write validation loss to disk
    with open(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'avg_val_losses.txt')), 'a') as valLossFile:
        valLossFile.write(timestamp)
        valLossFile.write('\n')
        valLossFile.writelines(str(avg_val_losses))

    # Write training loss to disk
    with open(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'avg_train_losses.txt')), 'a') as trainLossFile:
        trainLossFile.write(timestamp)
        trainLossFile.write('\n')
        trainLossFile.writelines(str(avg_train_losses))
    
    # Write training accuracy to disk
    with open(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'train_accuracy.txt')), 'a') as val1File:
        val1File.write(timestamp)
        val1File.write('\n')
        val1File.writelines(str(train_accuracies))
    
    # Write validation accuracy to disk
    with open(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'validation_accuracy.txt')), 'a') as val2File:
        val2File.write(timestamp)
        val2File.write('\n')
        val2File.writelines(str(val_accuracies))
    
    # Write test accuracy to disk
    with open(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'test_accuracy.txt')), 'a') as val3File:
        val3File.write(timestamp)
        val3File.write('\n')
        val3File.writelines(str(test_accuracies))
    
    print("Final test accuracy:", acc)
    
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
    plt.savefig(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'loss_plot.png')))

def plot_accuracy(acc_train, acc_val, acc_test, num_epochs):
    plt.clf()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b', label='validation accuracy')
    plt.plot(epochs, acc_test, 'r', label='Test accuracy')
    plt.title('Training, validation, test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Write loss plot to disk
    plt.savefig(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'acc_plot.png')))
    
# ------------------------------------------- LSTM TRAINING -------------------------------------------

def loss_fn(model, loss, device, data, target):
    data = data.to(device)
    target = target.to(device)
    prediction = model.forward(data)
    prediction = torch.squeeze(prediction, dim=1)

    return loss(prediction, target)

def trainModel(train_data, train_targets, test_data, test_targets, num_data, num_test_data, corpus, num_epochs = 150, batch_size = 1, val_size = 0.05):
    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    patience = 20
    min_epochs = 5
    epochs_no_improve = 0
    early_stop = False
    min_val_loss = np.inf

    val_split = num_data - math.floor(num_data * val_size)
    
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    # Congifuring the model
    if (FLAGS.loss == 'CTCLoss'):
        loss = nn.CTCLoss()
    else:
        loss = nn.CrossEntropyLoss()
    model = LSTM(FLAGS.num_ceps, corpus.get_phones_len(), size_hidden_layers=100)
    model.to(device)
    
    # Configuring the Optimizer (ADAptive Moments)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    bar = tqdm(range(num_epochs))

    for epoch in bar:

        for i in range(0, val_split):
            sample = train_data[i].type(torch.FloatTensor)
            target = train_targets[i].type(torch.LongTensor)
            sample = torch.reshape(sample, (sample.shape[0], 1, sample.shape[1]))
            optimizer.zero_grad()
            loss_val = loss_fn(model, loss, device, sample, target)
            train_losses.append(loss_val.item())
            loss_val.backward()
            optimizer.step()

        avg_train_losses.append(np.average(train_losses))

        model.eval()
        for i in range(val_split, num_data):
            sample = train_data[i].type(torch.FloatTensor)
            target = train_targets[i].type(torch.LongTensor)
            sample = torch.reshape(sample, (sample.shape[0], 1, sample.shape[1]))
            loss_val = loss_fn(model, loss, device, sample, target)
            val_losses.append(loss_val.item())
            
            if loss_val < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = loss_val
                torch.save(model.state_dict(), '{}/checkpoints/epoch{}earlystop{}'.format(FLAGS.results_save_dir, epoch, FLAGS.name))
            else:
                epochs_no_improve += 1
            if epoch > min_epochs and epochs_no_improve == patience:
                print("Early stopping!")
                early_stop = True
                break
            else:
                continue
                
        avg_val_loss = np.average(val_losses)
        avg_val_losses.append(avg_val_loss)
        model.train()
        
        accuracy1 = testModel(train_data[0:val_split], train_targets[0:val_split], val_split, model)
        accuracy2 = testModel(train_data[val_split:len(train_data)], train_targets[val_split:len(train_data)], (len(train_data) - val_split), model)
        accuracy3 = testModel(test_data, test_targets, num_test_data, model)
        train_accuracies.append(accuracy1)
        val_accuracies.append(accuracy2)
        test_accuracies.append(accuracy3)

        bar.set_description(
            'train_loss {:.3f}; val_loss {:.3f}, train_accuracy {:.3f}, val_accuracy {:.3f}, test_accuracy {:.3f}'.format(
                avg_train_losses[-1], avg_val_losses[-1], train_accuracies[-1], val_accuracies[-1], test_accuracies[-1])
        )

        if epoch > 0 and epoch % 10 == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_losses[-1],
                'train_loss': avg_train_losses[-1]
            }
            timestamp = datetime.now()
            name = 'checkpoint' + str(timestamp) + 'epoch {}.pt'.format(epoch)
            path = os.path.join(os.path.abspath(os.path.join(
                FLAGS.results_save_dir, 'checkpoints')), name)
            torch.save(checkpoint_dict, path)

    return model, avg_val_losses, avg_train_losses, train_accuracies, val_accuracies, test_accuracies

def testModel(test_data, test_targets, num_data, model):
    correct = 0
    total = 0
    
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    model.eval()
    
    bar = tqdm(range(num_data))

    for i in bar:
        sample = test_data[i]
        target = test_targets[i]
        sample = torch.reshape(sample, (sample.shape[0], 1, sample.shape[1]))
        sample = sample.to(device)
        target = target.to(device)
        prediction = model.forward(sample)
        prediction = torch.squeeze(prediction, dim=1)
        _, prediction_label = torch.max(prediction, dim=1)
        correct += (prediction_label == target).sum()
        total += target.shape[0]
    accuracy = correct / total * 100
    
    model.train()

    return accuracy
    
# ------------------------------------------- DATA LOADING -------------------------------------------

def getMFCCFeatures(dataset, zeropad = False, oneTensor = False):
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
            if(max_frames < audio.shape[0]):
                max_frames = audio.shape[0]
            if(zeropad == True):
                features.append(audio)
            else:
                tensors.append(torch.tensor(audio.tolist(), dtype=torch.float))
    
    if(zeropad == True):
        # zero-padding for equal length
        for i in range(len(dataset)):
            audio_new = np.zeros((max_frames, features[i].shape[1]))
            audio_new[0:features[i].shape[0],:] = features[i]
            tensors.append(torch.tensor(audio_new.tolist(), dtype=torch.float))
    
    if(oneTensor == True):
        whole = tensors[0].numpy()
        for i in range(1, len(dataset)):
            whole = np.concatenate((whole, tensors[i].numpy()), axis = 0)
        tensors = torch.tensor(whole.tolist(), dtype = torch.float)

    return tensors, max_frames

def getTargetPhonemes(dataset, max_frames, corpus, zeropad = False, oneTensor = False, mode = "indices"):
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
            if(mode == "indices"):
                # using only one phoneme explicitly imposed --> first phoneme index
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_id = corpus.get_phones_ID(the_one_phoneme)
                sample_targets.append(the_one_phoneme_id)
            elif(mode == "percentages"):
                if(len(phoneme_list[j]) == 1):
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
            elif(mode == "onehot"):
                # using only one phoneme explicitly imposed --> one hot encoding
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_id = corpus.get_phones_ID(the_one_phoneme)
                the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[0]
                sample_targets.append(the_one_phoneme_one_hot)
            else:
                print("Wrong mode!")
                break
        
        if(zeropad == True):
            for j in range(len(phoneme_list), max_frames):
                # adding a silence target for the extra frames (zero-padded additions)
                silence = corpus.get_silence()
                if(mode == "percentages" or mode == "onehot"):
                    silence_one_hot = corpus.phones_to_onehot([silence])[0]
                    sample_targets.append(silence_one_hot)
                elif(mode == "indices"):
                    silence_id = corpus.get_phones_ID(silence)
                    sample_targets.append(silence_id)
                
        tensors.append(torch.tensor(sample_targets, dtype=torch.long))
        
    if(oneTensor == True):
        whole = tensors[0].numpy()
        for i in range(1, len(dataset)):
            whole = np.concatenate((whole, tensors[i].numpy()), axis = 0)
        tensors = torch.tensor(whole.tolist(), dtype = torch.long)

    return tensors

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

if __name__ == '__main__':
    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')
    flags.DEFINE_integer('frame_len', 20, 'Frame length in ms')
    flags.DEFINE_integer('frame_shift', 10, 'frame shift in ms')

    flags.DEFINE_integer('num_epochs', 60, 'Number of epochs')
    flags.DEFINE_float('lr', 0.001, 'Learning rate')
    flags.DEFINE_string('dataset_root_dir', 'timit',
                        'The path to the dataset root directory')
    flags.DEFINE_string('results_save_dir', 'results',
                        'The path to the directory where all the results are saved')
    flags.DEFINE_integer('hidden', 100, 'number of nodes in each LSTM layer')
    flags.DEFINE_string('name', 'vanillaLSTMfullylabeled.pth', 'name of model')
    flags.DEFINE_string('loss', 'CrossEntropyLoss',
                        'The name of loss function')

    app.run(main)
