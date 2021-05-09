import os.path
import test
import torch.utils.data
import torch.nn as nn
from absl import app
import numpy as np
from datetime import datetime


from tqdm import tqdm

from models.lstm import LSTM

from absl import flags
from datasets.TIMITdataset import TimitDataset

FLAGS = flags.FLAGS


def main(args):
    root_dir = 'timit'

    dataset = TimitDataset(csv_file='train_data.csv', root_dir=root_dir,
                           pre_epmh=FLAGS.preemphasis_coefficient,
                           num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                           frame_shift=FLAGS.frame_shift)
    '''important before training a model set model name so checkpoints 
    and fully trained model gets saved with correct name'''
    model_name = 'vanillaLSTMfullylabeled.pth'
    save_folder = os.path.abspath('trained_models')
    save_path = os.path.join(save_folder, model_name)

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    model, avg_val_losses, avg_train_losses = train(dataset, num_epochs= 1)
    os.makedirs(save_folder, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    timestamp = str(datetime.now())
    with open('avg_val_losses.txt', 'a') as valLossFile:
        valLossFile.write(timestamp)
        valLossFile.writelines(str(avg_val_losses))
    with open('avg_train_losses.txt', 'a') as trainLossFile:
        trainLossFile.write(timestamp)
        trainLossFile.writelines(str(avg_train_losses))

    test_data = TimitDataset(csv_file='test_data.csv', root_dir=root_dir,
                           pre_epmh=FLAGS.preemphasis_coefficient,
                           num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                           frame_shift=FLAGS.frame_shift)

    correct = 0
    total = 0
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=3)
    with torch.no_grad():
        for point in tqdm(test_loader):
            sample, target = point
            sample = sample.to(device)
            target = target.to(device)
            output = model.forward(sample)
            prediction = torch.max(output, dim=0)
            correct += (prediction == target).float().sum()
            total += target.shape[0]
    accuracy = correct / total * 100

    with open('test_accuracy.txt', 'a') as test_accuracy:
        test_accuracy.write(timestamp, str(accuracy))
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=3, shuffle=True)
    '''if batch_size > 1:
        train_loader = torch.nn.utils.rnn.pad_packed_sequence(train_loader)'''
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=2)
    #val_loader = torch.nn.utils.rnn.pad_packed_sequence(val_loader)

    loss = nn.CrossEntropyLoss()
    model = LSTM(FLAGS.num_ceps, dataset.num_labels, size_hidden_layers=100)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    #early_stop = EarlyStopping(patience=patience, verbose=True)
    for epoch in tqdm(range(num_epochs), desc='training epochs'):

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

        if epoch > 10:
            model.eval()
            for data, target in tqdm(val_loader):
                data = data.to(device)
                target = target.to(device)
                output = model.forward(data)
                val_loss = loss(output.squeeze(), target.squeeze())
                val_losses.append(val_loss.item())
            avg_val_loss = np.average(val_losses)
            avg_val_losses.append(avg_val_loss)

        if epoch > 0 and epoch % 10 == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_losses[-1],
                'train_loss': avg_train_losses[-1]
            }
            name = 'checkpoint epoch {}.pt'.format(epoch)
            path = '../checkpoints/' + name
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
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=2)
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


            #batch.set_postfix(loss.item())



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