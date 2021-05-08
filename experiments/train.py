import os.path
from SemiSupervisedLearningASR import test
import torch.utils.data
import torch.nn as nn
from absl import app
import numpy as np


from tqdm import tqdm

from SemiSupervisedLearningASR.models.lstm import LSTM

from absl import flags
from SemiSupervisedLearningASR.datasets.TIMITdataset import TimitDataset

FLAGS = flags.FLAGS


def main(args):
    root_dir = os.path.abspath('timit')
    print(root_dir)
    dataset = TimitDataset(csv_file='train_data.csv', root_dir=root_dir,
                           pre_epmh=FLAGS.preemphasis_coefficient,
                           num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                           frame_shift=FLAGS.frame_shift)
    '''important before training a model set model name so checkpoints 
    and fully trained model gets saved with correct name'''
    model_name = 'vanillaLSTMfullylabeled.pth'
    save_folder = os.path.abspath('trained_models')
    save_path = os.path.join(save_folder, model_name)
    model, avg_val_losses, avg_train_losses = train(dataset, num_epochs= 20)

    torch.save(model.state_dict(), save_path)

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
            output = model.forward(sample)
            prediction = torch.max(output, dim=0)
            correct += (prediction == target).float().sum()
            total += target.shape[0]
    accuracy = correct / total * 100
    print(accuracy)



def train(train_set, val_set, num_epochs, batch_size=1):
    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    patience = 20

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=3, shuffle=True)
    if batch_size > 1:
        train_loader = torch.nn.utils.rnn.pad_packed_sequence(train_loader)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=12, num_workers=2)
    val_loader = torch.nn.utils.rnn.pad_packed_sequence(val_loader)

    loss = nn.CrossEntropyLoss()
    model = LSTM(FLAGS.num_ceps, dataset.num_labels, size_hidden_layers=100)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    early_stop = EarlyStopping(patience=patience, verbose=True)
    for epoch in tqdm(range(num_epochs), desc='training epochs'):

        for batch in tqdm(train_loader, desc="training batches"):
            sample, target = batch
            sample.to(device)
            target.to(device)
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
                output = model.forward(data)
                val_loss = loss(output.squeeze(), target.squeeze())
                val_losses.append(val_loss.item())
            avg_val_loss = np.average(val_losses)
            avg_val_losses.append(avg_val_loss)

            early_stop(avg_val_loss, model)
            if early_stop.early_stop:
                print("Early stopping")
                model.load_state_dict(torch.load('checkpoint.pt'))
                break

    return model, avg_val_losses, avg_train_losses

def validate(val_set, model):
    correct = 0
    total = 0
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=12, num_workers=2)
    val_loader = torch.nn.utils.rnn.pad_packed_sequence(val_loader)

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