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

from models.lstm import LSTMModel
from models.lstm1 import LSTM

from absl import flags
from datasets.TIMITdataset import TimitDataset

from mean_teacher import MeanTeacher
from baseline import Baseline

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
        dataset, method = FLAGS.method, num_epochs=FLAGS.num_epochs)

    torch.save(model.state_dict(), save_path)

    np.savetxt(os.path.abspath(os.path.join(FLAGS.results_save_dir,
               'avg_val_losses.txt')), np.asarray(avg_val_losses), delimiter=',')
    np.savetxt(os.path.abspath(os.path.join(FLAGS.results_save_dir,
               'avg_train_losses.txt')), np.asarray(avg_train_losses), delimiter=',')

    plt.plot(avg_train_losses)
    plt.plot(avg_val_losses)
    plt.savefig(os.path.abspath(os.path.join(
        FLAGS.results_save_dir, 'loss_plot.png')))

    test_data = TimitDataset(csv_file='test_data.csv', root_dir=FLAGS.dataset_root_dir,
                             pre_epmh=FLAGS.preemphasis_coefficient,
                             num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len,
                             frame_shift=FLAGS.frame_shift)

    accuracy = test.test_model(model, test_data)

    print(accuracy)


def loss_fn(model, loss, device, data, target):
    data = data.to(device)
    target = target.to(device)
    target = torch.squeeze(target, dim=0)

    output_teacher, output_student = mean_teacher.forward(data)

    output_teacher, output_student = torch.squeeze(
        output_teacher, dim=0), torch.squeeze(output_student, dim=0)

    prediction_2 = torch.squeeze(prediction, dim=0)

    prediction = model.forward(data)

    prediction_2 = torch.squeeze(prediction, dim=0)
    target_2 = torch.squeeze(target, dim=0)

    return loss(prediction_2, target_2)


def train(dataset, num_epochs, method, batch_size=1):
    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    accuracies = []
    patience = 3
    min_epochs = 5
    epochs_no_improve = 0
    early_stop = False
    min_val_loss = np.inf

    val_split = int(len(dataset)*FLAGS.ratio_validation_data)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - val_split, val_split],
                                                         generator=torch.Generator().manual_seed(15))

    unlabeled_split = int(len(train_data)*(1 - FLAGS.ratio_labeled_data))
    labeled_train_data, unlabeled_train_data = torch.utils.data.random_split(train_data, [len(train_data) - unlabeled_split, unlabeled_split],
                                                                             generator=torch.Generator().manual_seed(15))

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda:0')
    else:
        print('using cpu')
        device = torch.device('cpu')

    labeled_train_loader = torch.utils.data.DataLoader(
        labeled_train_data, batch_size=batch_size, num_workers=3, shuffle=True)

    unlabeled_train_loader = torch.utils.data.DataLoader(
        unlabeled_train_data, batch_size=batch_size, num_workers=3, shuffle=True)
    '''if batch_size > 1:
        train_loader = torch.nn.utils.rnn.pad_packed_sequence(train_loader)'''
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, num_workers=2)
    #val_loader = torch.nn.utils.rnn.pad_packed_sequence(val_loader)

    # Add CTC Loss
    if (FLAGS.loss == 'CrossEntropyLoss'):
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.CrossEntropyLoss()

    # model = LSTMModel(input_dim=13, hidden_dim=500,
    #                  layer_dim=1, output_dim=dataset.num_labels)

    #model = LSTM(FLAGS.num_ceps, dataset.num_labels, size_hidden_layers=100)
    if method == 'mean_teacher':
        model = MeanTeacher(FLAGS.num_ceps, dataset.num_labels,
                        size_hidden_layers=100, ema_decay=0.999)

    elif method == 'baseline':
        model = Baseline(FLAGS.num_ceps, dataset.num_labels,
                        size_hidden_layers=100)
    else:
        raise Exception('Wrong flag for method')

    model.to(device)
    optimizer = model.get_optimizer()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(
    #    0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #early_stop = EarlyStopping(patience=patience, verbose=True)


    bar = tqdm(range(num_epochs))

    for epoch in bar:

        l_iter = iter(labeled_train_loader)

        for u_batch in unlabeled_train_loader:
            u_data, _ = u_batch
            u_data = u_data.to(device)

            try:
                l_data, target, = next(l_iter)
            except StopIteration:
                l_iter = iter(labeled_train_loader)
                l_data, target, = next(l_iter)

            l_data, target = l_data.to(device), target.to(device)
        
            loss_value = model.train_step(device, u_data, l_data, target)

            train_losses.append(loss_value.item())

        avg_train_losses.append(np.average(train_losses))

        model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = torch.squeeze(target, dim=0)

            output = model.forward(data)

            loss_value = model.loss_fn(output, target)

            val_losses.append(loss_value.item())
#             print('local value')
#             print(val_losses)
#             print('minimum value loss')
#             print(min_val_loss)
            if loss_value < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = loss_value
#                 print("NONE IMPROVEMENT")
                torch.save(model.state_dict(
                ), '{}/checkpoints/epoch{}earlystop{}'.format(FLAGS.results_save_dir, epoch, FLAGS.name))
            else:
                epochs_no_improve += 1
#                 print("IMPROVE epochs ")
            if epoch > min_epochs and epochs_no_improve == patience:
                print("Early stopping!")
                early_stop = True
                break
            else:
                continue

        avg_val_loss = np.average(val_losses)
        avg_val_losses.append(avg_val_loss)

        accuracy = validate(val_loader, model, device)

        accuracies.append(accuracy)

        model.train()

        bar.set_description(
            'train_loss {:.3f}; val_loss {:.3f}, val_accuracy {:.3f}'.format(
                avg_train_losses[-1], avg_val_losses[-1], accuracies[-1])
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
        if early_stop:
            print("Early stopped after {} epochs".format(epoch))
            break
    return model, avg_val_losses, avg_train_losses


def validate(val_loader, model, device):
    correct = 0
    total = 0
    model.eval()
    for point in val_loader:
        sample, target = point
        sample = sample.to(device)
        target = target.to(device)

        output = model.forward(sample)
        _, prediction = torch.max(output, dim=1)

        correct += (prediction == target).sum()
        total += target.shape[1]
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
    flags.DEFINE_string('loss', 'CrossEntropyLoss',
                        'The name of loss function')
    flags.DEFINE_enum('method', 'baseline', ['baseline', 'mean_teacher'], 'The method: baseline, mean_teacher.')
    flags.DEFINE_float('ratio_labeled_data', 0.1, 'Ratio of unlabeled data.')
    flags.DEFINE_float('ratio_validation_data', 0.15, 'Ratio of validation data.')

    app.run(main)
