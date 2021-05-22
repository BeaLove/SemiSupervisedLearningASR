import copy
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

from baseline import Baseline
from mean_teacher import MeanTeacher

from absl import logging
logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS


def main(argv):
    """ This is our main method.
    """
    del argv  # Unused.

    set_seeds(0)

    logging.info("Method: {}".format(FLAGS.method))
    logging.info("labeled_p: {}".format(FLAGS.labeled_p))
    logging.info("num_epochs: {}".format(FLAGS.num_epochs))
    logging.info("batch_size: {}".format(FLAGS.batch_size))
    logging.info("num_hidden: {}".format(FLAGS.num_hidden))
    logging.info("num_layers: {}".format(FLAGS.num_layers))
    logging.info("dropout: {}".format(FLAGS.dropout))
    logging.info("optimizer: {}".format(FLAGS.optimizer))
    logging.info("lr: {}".format(FLAGS.lr))

    # Initialize a Corpus object
    example_file_dir = "/data/TRAIN/DR1/FCJF0/SA1"  # SA1.wav.WAV
    dataset_dir = 'timit'
    corpus = Corpus(dataset_dir, example_file_dir)  # TIMIT corpus
    phonemes = corpus.get_phonemes()  # List of phonemes
    targets = len(phonemes)  # Number of categories

    # Load the TIMIT dataset
    train_dataset = TimitDataset(csv_file='train_data.csv',
                                 root_dir=dataset_dir,
                                 corpus=corpus,
                                 transform=MFCC(n_fft=FLAGS.n_fft,
                                                preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                                num_ceps=FLAGS.num_ceps),
                                 transcription=Phonemes(n_fft=FLAGS.n_fft,
                                                        preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                                        num_ceps=FLAGS.num_ceps,
                                                        corpus=corpus))

    test_dataset = TimitDataset(csv_file='test_data.csv',
                                root_dir=dataset_dir,
                                corpus=corpus,
                                transform=MFCC(n_fft=FLAGS.n_fft,
                                               preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                               num_ceps=FLAGS.num_ceps),
                                transcription=Phonemes(n_fft=FLAGS.n_fft,
                                                       preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                                       num_ceps=FLAGS.num_ceps,
                                                       corpus=corpus))

    if os.path.isdir('tensors') == False:
        logging.info(
            "Generate MFCC coefficients and save it in the directory tensors")
        makedirs('tensors')

        # Get the MFCC coefficients
        train_data, max_len = getMFCCFeatures(train_dataset, oneTensor=False)
        test_data, max_len_test = getMFCCFeatures(
            test_dataset, oneTensor=False)

        # Get the phonemes per frame
        train_targets = getTargetPhonemes(
            train_dataset, max_len, corpus, oneTensor=False, mode="indices")
        test_targets = getTargetPhonemes(
            test_dataset, max_len, corpus, oneTensor=False, mode="indices")

        writedirs(train_data, "train_data.pt")
        writedirs(test_data, "test_data.pt")
        writedirs(train_targets, "train_targets.pt")
        writedirs(test_targets, "test_targets.pt")
        writeelem(max_len, "max_len.pt")
        writeelem(max_len_test, "max_len_test.pt")

    else:
        logging.info("Load MFCC coefficients from the directory tensors")
        # Get the MFCC coefficients
        train_data = readdirs("train_data.pt", len(train_dataset))
        max_len = readelem("max_len.pt")
        test_data = readdirs("test_data.pt", len(test_dataset))
        max_len_test = readelem("max_len_test.pt")

        # Get the phonemes per frame
        train_targets = readdirs("train_targets.pt", len(train_dataset))
        test_targets = readdirs("test_targets.pt", len(test_dataset))

    # Create directories
    makedirs(FLAGS.results_save_dir)
    makedirs(os.path.join(FLAGS.results_save_dir, 'checkpoints'))

    # Configure model
    model_name = FLAGS.name
    save_folder = os.path.abspath(FLAGS.results_save_dir)
    save_path = os.path.join(save_folder, model_name)

    # Train model
    epochNum = FLAGS.num_epochs
    model, avg_val_losses, avg_train_losses, train_accuracies, val_accuracies, test_accuracies = trainModel(
        train_data, train_targets, test_data, test_targets, len(train_dataset), len(test_dataset), corpus, num_epochs=epochNum)
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


def writedirs(mylist, filename):
    for i in range(0, len(mylist)):
        torch.save(mylist[i], ("tensors/" + str(i) + filename))


def readdirs(filename, lim):
    newlist = []
    for i in tqdm(range(0, lim)):
        decomc = torch.load(("tensors/" + str(i) + filename))
        newlist.append(decomc)

    return newlist


def writeelem(element, filename):
    ten = torch.from_numpy(np.array([element]))

    torch.save(ten, filename)


def readelem(filename):

    decomc = torch.load(filename)
    num = decomc.numpy()[0]
    return num

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
    plt.savefig(os.path.abspath(os.path.join(
        FLAGS.results_save_dir, 'loss_plot.png')))


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
    plt.savefig(os.path.abspath(os.path.join(
        FLAGS.results_save_dir, 'acc_plot.png')))

# ------------------------------------------- LSTM TRAINING -------------------------------------------


def loss_fn(model, loss, device, data, target):
    data = data.to(device)
    target = target.to(device)
    prediction = model.forward(data)
    prediction = torch.squeeze(prediction, dim=1)

    return loss(prediction, target)


def get_targets(targets, p):
    labeled = np.random.binomial(1, p, len(targets)) > 0

    for i, trail in enumerate(labeled):
        if trail == False:
            targets[i] = None

    return targets


def trainModel(train_data, train_targets, test_data, test_targets, num_data, num_test_data, corpus, num_epochs=150, batch_size=1, val_size=0.05, labeled_size=0.1):
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

    full_train_targets = copy.deepcopy(train_targets)

    val_split = num_data - math.floor(num_data * val_size)
    labeled_split = val_split - math.floor(val_split * labeled_size)

    if torch.cuda.is_available():
        logging.info("Using Cuda")
        device = torch.device('cuda:0')
    else:
        logging.info("Using cpu")
        device = torch.device('cpu')

    # Congifuring the model
    if (FLAGS.loss == 'CTCLoss'):
        loss = nn.CTCLoss()
    else:
        loss = nn.CrossEntropyLoss()

    if FLAGS.method == 'mean_teacher':
        # need to check this. add flag
        consistency_rampup = len(train_data) * 5

        model = MeanTeacher(mfccs=FLAGS.num_ceps, output_phonemes=corpus.get_phones_len(),
                            units_per_layer=FLAGS.num_hidden, num_layers=FLAGS.num_layers,
                            dropout=FLAGS.dropout, optimizer=FLAGS.optimizer, lr=FLAGS.lr,
                            max_steps=10000, ema_decay=0.999, consistency_weight=1.0)

        train_targets[0:val_split] = get_targets(
            train_targets[0:val_split], p=FLAGS.labeled_p)

    elif FLAGS.method == 'baseline':
        model = Baseline(loss,
                         mfccs=FLAGS.num_ceps, output_phonemes=corpus.get_phones_len(),
                         units_per_layer=FLAGS.num_hidden, num_layers=FLAGS.num_layers,
                         dropout=FLAGS.dropout, optimizer=FLAGS.optimizer, lr=FLAGS.lr)
        train_targets[0:val_split] = get_targets(
            train_targets[0:val_split], p=FLAGS.labeled_p)
    else:
        raise Exception('Wrong flag for method')
    model.to(device)

    optimizer = model.get_optimizer()

    count_labeled_samples = 0
    count_unlabeled_samples = 0

    for t in train_targets:
        if not(t is None):
            count_labeled_samples += 1
        else:
            count_unlabeled_samples += 1

    logging.info("Labeled samples: {}".format(count_labeled_samples))
    logging.info("Unlabeled samples: {}".format(count_unlabeled_samples))

    bar = tqdm(range(num_epochs))
    for epoch in bar:

        if FLAGS.method == 'mean_teacher':
            model.update_rampup(epoch, FLAGS.consistency_rampup)

        for i in range(0, val_split, FLAGS.batch_size):

            loss_val = 0
            optimizer.zero_grad()

            for batch_idx in range(i, min(val_split, i + FLAGS.batch_size)):  # pen and papper

                sample, target = train_data[batch_idx], train_targets[batch_idx]

                sample = sample.type(torch.FloatTensor)
                sample = torch.reshape(
                    sample, (sample.shape[0], 1, sample.shape[1]))

                if not(target is None):
                    target = target.type(torch.LongTensor)

                result = model.loss_fn(device, sample, target)
                loss_val += result

            model.train_step(loss_val)

            train_losses.append(loss_val.item())

        avg_train_losses.append(np.average(train_losses))

        model.eval()
        for i in range(val_split, num_data):
            sample = train_data[i].type(torch.FloatTensor)
            target = train_targets[i].type(torch.LongTensor)
            sample = torch.reshape(
                sample, (sample.shape[0], 1, sample.shape[1]))

            loss_val = model.loss_fn(device, sample, target)
            # loss_val = loss_fn(model, loss, device, sample, target)
            val_losses.append(loss_val.item())

            if loss_val < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = loss_val
                torch.save(model.state_dict(
                ), '{}/checkpoints/epoch{}earlystop{}'.format(FLAGS.results_save_dir, epoch, FLAGS.name))
            else:
                epochs_no_improve += 1
            if epoch > min_epochs and epochs_no_improve == patience:
                logging.info("Early stopping!")
                early_stop = True
                break
            else:
                continue

        avg_val_loss = np.average(val_losses)
        avg_val_losses.append(avg_val_loss)
        model.train()

        accuracy1 = testModel(
            train_data[0:val_split], full_train_targets[0:val_split], val_split, model)
        accuracy2 = testModel(train_data[val_split:len(train_data)], full_train_targets[val_split:len(
            train_data)], (len(train_data) - val_split), model)
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
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.eval()

    # bar = tqdm(range(num_data))

    for i in range(num_data):
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


def getMFCCFeatures(dataset, zeropad=False, oneTensor=False):
    """ This method computes the MFCC coefficients per frame.
         When frames are less than the maximum amount does zero-padding.
         @returns tensors of MFCC coefficients of the same length
    """
    features = []
    tensors = []
    max_frames = -1

    for i in tqdm(range(len(dataset))):
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
            audio_new[0:features[i].shape[0], :] = features[i]
            tensors.append(torch.tensor(audio_new.tolist(), dtype=torch.float))

    if(oneTensor == True):
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

    for i in tqdm(range(len(dataset))):
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
                    the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[
                        0]
                    sample_targets.append(the_one_phoneme_one_hot)
                else:
                    # more than one phonemes in this frame --> probabilities
                    complex_one_hot = np.zeros(
                        corpus.get_phones_len()).tolist()
                    for k in range(len(phoneme_list[j])):
                        the_kth_phoneme = phoneme_list[j][k][2]
                        the_kth_phoneme_ID = corpus.get_phones_ID(
                            the_kth_phoneme)
                        the_kth_phoneme_percentage = phoneme_list[j][k][3]
                        complex_one_hot[the_kth_phoneme_ID] = the_kth_phoneme_percentage
                    sample_targets.append(complex_one_hot)
            elif(mode == "onehot"):
                # using only one phoneme explicitly imposed --> one hot encoding
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_id = corpus.get_phones_ID(the_one_phoneme)
                the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[
                    0]
                sample_targets.append(the_one_phoneme_one_hot)
            else:
                logging.info("Wrong mode")
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
            whole = np.concatenate((whole, tensors[i].numpy()), axis=0)
        tensors = torch.tensor(whole.tolist(), dtype=torch.long)

    return tensors


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


if __name__ == '__main__':
    flags.DEFINE_string('dataset_root_dir', 'timit',
                        'The path to the dataset root directory')
    flags.DEFINE_string('results_save_dir', 'results',
                        'The path to the directory where all the results are saved')

    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')
    flags.DEFINE_integer('frame_len', 20, 'Frame length in ms')
    flags.DEFINE_integer('frame_shift', 10, 'frame shift in ms')

    flags.DEFINE_integer('num_epochs', 2, 'Number of epochs')
    flags.DEFINE_float('lr', 0.001, 'Learning rate')

    flags.DEFINE_string('name', 'vanillaLSTMfullylabeled.pth', 'name of model')
    flags.DEFINE_enum('loss', 'CrossEntropyLoss', [
                      'CrossEntropyLoss', 'CTCLoss'], 'The name of loss function')

    flags.DEFINE_float('labeled_p', 0.95, 'Labeled percentage of data')
    flags.DEFINE_integer('batch_size', 1, 'The batch size')
    flags.DEFINE_enum('method', 'baseline', [
                      'baseline', 'mean_teacher'], 'The method: baseline, mean_teacher.')
    flags.DEFINE_float('consistency_weight', 1.0,
                       'The consistency weight for the mean teacher loss.')
    flags.DEFINE_integer('consistency_rampup', 5,
                         'The rampup for the consistency weight')
    flags.DEFINE_enum('optimizer', 'AdamNormGrad', [
                      'Adam', 'AdamNormGrad'], 'The optimizer: Adam, AdamNormGrad (Adam with normalizing gradients)')
    flags.DEFINE_integer('num_hidden', 100, 'Size of hidden layer.')
    flags.DEFINE_integer('num_layers', 1, 'The number of layers.')
    flags.DEFINE_float(
        'dropout', 0.1, 'If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout')

    app.run(main)
