import os.path
import test
import torch.utils.data
import torch.nn as nn
from absl import app
import numpy as np
from datetime import datetime
from test import test_model
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.lstm import LSTM

from absl import flags
from datasets.TIMITdataset import TimitDataset

FLAGS = flags.FLAGS


def makedirs(path):
    Path(path).mkdir(parents=False, exist_ok=True)


def main(args):
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    makedirs(FLAGS.results_save_dir)
    makedirs(os.path.join(FLAGS.results_save_dir, 'checkpoints'))

    model_name = FLAGS.name

    save_folder = os.path.abspath(FLAGS.results_save_dir)
    save_path = os.path.join(save_folder, model_name)

    dataset = TimitDataset(csv_file='train_data.csv', root_dir=FLAGS.dataset_root_dir,
                           pre_epmh=FLAGS.preemphasis_coefficient,
                           num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                           frame_shift=FLAGS.frame_shift)
    '''important before training a model set model name so checkpoints 
    and fully trained model gets saved with correct name'''

    model, avg_val_losses, avg_train_losses = train(
        dataset, num_epochs=FLAGS.num_epochs)

    torch.save(model.state_dict(), save_path)

    timestamp = str(datetime.now())
    with open(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'avg_val_losses.txt')), 'a') as valLossFile:
        valLossFile.write(timestamp)
        valLossFile.write('\n')
        valLossFile.writelines(str(avg_val_losses))

    with open(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'avg_train_losses.txt')), 'a') as trainLossFile:
        trainLossFile.write(timestamp)
        trainLossFile.write('\n')
        trainLossFile.writelines(str(avg_train_losses))

    plt.plot(avg_train_losses)
    plt.plot(avg_val_losses)
    plt.savefig(os.path.abspath(os.path.join(FLAGS.results_save_dir, 'loss_plot.png')))

    test_data = TimitDataset(csv_file='test_data.csv', root_dir=FLAGS.dataset_root_dir,
                             pre_epmh=FLAGS.preemphasis_coefficient,
                             num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                             frame_shift=FLAGS.frame_shift)

    accuracy = test.test_model(model, test_data)

    print(accuracy)


def train(dataset, num_epochs, batch_size=1):
    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    patience = 20

    val_split = int(len(dataset)*0.15)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - val_split, val_split],
                                                         generator=torch.Generator().manual_seed(15))
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=3, shuffle=True)
    '''if batch_size > 1:
        train_loader = torch.nn.utils.rnn.pad_packed_sequence(train_loader)'''
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, num_workers=2)
    #val_loader = torch.nn.utils.rnn.pad_packed_sequence(val_loader)

    loss = nn.CrossEntropyLoss()
    model = LSTM(FLAGS.num_ceps, dataset.num_labels, size_hidden_layers=100)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #early_stop = EarlyStopping(patience=patience, verbose=True)

    bar = tqdm(range(num_epochs))

    for epoch in bar:

        for batch in train_loader:
            sample, target = batch
            sample = sample.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            prediction = model.forward(sample)
            loss_val = loss(prediction.squeeze(), target.squeeze())
            train_losses.append(loss_val.item())
            loss_val.backward()
            optimizer.step()

        avg_train_losses.append(np.average(train_losses))

        model.eval()
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model.forward(data)
            val_loss = loss(output.squeeze(), target.squeeze())
            val_losses.append(val_loss.item())
        avg_val_loss = np.average(val_losses)
        avg_val_losses.append(avg_val_loss)
        model.train()

        bar.set_description(
            'train_loss {:.3f}; loss_loss {:.3f}'.format(
                avg_train_losses[-1], avg_val_losses[-1])
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

            '''early_stop(avg_val_loss, model)
            if early_stop.early_stop:
                print("Early stopping")
                model.load_state_dict(torch.load('checkpoint.pt'))
                break'''

    return model, avg_val_losses, avg_train_losses


def validate(val_set, model):
    correct = 0
    total = 0
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, num_workers=2)
    #val_loader = torch.nn.utils.rnn.pad_packed_sequence(val_loader)

    model.eval()
    for point in tqdm(val_loader):
        sample, target = point
        sample = torch.nn.utils.rnn.pad_packed_sequence(sample)
        output = model.forward(sample)
        prediction = torch.max(output, dim=0)
        correct += (prediction == target).float().sum()
        total += target.shape[0]
    accuracy = correct / total * 100

    return accuracy

    # batch.set_postfix(loss.item())


if __name__ == '__main__':

    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')
    flags.DEFINE_integer('frame_len', 20, 'Frame length in ms')
    flags.DEFINE_integer('frame_shift', 10, 'frame shift in ms')

    flags.DEFINE_integer('num_epochs', 2, 'Number of epochs')
    flags.DEFINE_float('lr', 0.001, 'Learning rate')
    flags.DEFINE_string('dataset_root_dir', 'timit',
                        'The path to the dataset root directory')
    flags.DEFINE_string('results_save_dir', 'results',
                        'The path to the directory where all the results are saved')
    flags.DEFINE_integer('hidden', 100, 'number of nodes in each LSTM layer')
    flags.DEFINE_string('name', 'vanillaLSTMfullylabeled.pth', 'name of model')


    app.run(main)
