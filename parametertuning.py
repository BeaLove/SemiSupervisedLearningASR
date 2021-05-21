from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from datasets.TIMITdataset import TimitDataset
from absl import flags
import matplotlib.pyplot as plt
from train import train, loss_fn, validate
from models.lstm1 import LSTM

FLAGS = flags.FLAGS

def main():
    data_dir = 'timit'
    gpus_per_trial = 0.5
    max_num_epochs = 100
    num_samples = 8
    config = {
        "num_layers" : tune.sample_from(lambda _: np.random.randint(1, 5)), #number of layers in the model
        "layer_size" : tune.choice([25, 50, 75, 100, 150]), #number of hidden nodes in each layer
        "lr": tune.loguniform(1e-4, 1e-1),
        #"dropout":                              #dropout probability
        #"batch_size": tune.choice([2, 4, 8, 16]) # batch size
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
        partial(train_params, data_dir= data_dir),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


def load_data(root_dir = "timit"):
    train_set = TimitDataset(csv_file='train_data.csv', root_dir=FLAGS.dataset_root_dir,
                           pre_epmh=FLAGS.preemphasis_coefficient,
                           num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                           frame_shift=FLAGS.frame_shift)
    test_set = TimitDataset(csv_file='test_data.csv', root_dir=FLAGS.dataset_root_dir,
                           pre_epmh=FLAGS.preemphasis_coefficient,
                           num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                           frame_shift=FLAGS.frame_shift)
    return train_set, test_set

def train_params(config, data_dir = 'timit'):

    train_set, test_set = load_data(data_dir)

    model = LSTM(FLAGS.num_ceps, output_phonemes=50, size_hidden_layers=config['layer_size'],
                 num_layers=config['num_layers'], dropout = config['dropout'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    model, avg_train_losses, avg_val_losses = train(train_set, model, num_epochs=100, batch_size=config['batch_size'])

    np.savetxt(os.path.abspath(os.path.join(FLAGS.results_save_dir,
                                            'avg_val_losses.txt')), np.asarray(avg_val_losses), delimiter=',')
    np.savetxt(os.path.abspath(os.path.join(FLAGS.results_save_dir,
                                            'avg_train_losses.txt')), np.asarray(avg_train_losses), delimiter=',')

    plt.plot(avg_train_losses)
    plt.plot(avg_val_losses)
    plt.savefig(os.path.abspath(os.path.join(
        FLAGS.results_save_dir, 'loss_plot.png')))



if __name__ == '__main__':

    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')
    flags.DEFINE_integer('frame_len', 20, 'Frame length in ms')
    flags.DEFINE_integer('frame_shift', 10, 'frame shift in ms')

    flags.DEFINE_integer('num_epochs', 100, 'Number of epochs')
    flags.DEFINE_float('lr', 0.001, 'Learning rate')
    flags.DEFINE_string('dataset_root_dir', 'timit',
                        'The path to the dataset root directory')
    flags.DEFINE_string('results_save_dir', 'results',
                        'The path to the directory where all the results are saved')
    flags.DEFINE_integer('hidden', 100, 'number of nodes in each LSTM layer')
    flags.DEFINE_string('name', 'vanillaLSTMfullylabeled.pth', 'name of model')
    flags.DEFINE_string('loss', 'CrossEntropyLoss',
                        'The name of loss function')

    main()