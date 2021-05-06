import os.path

import torch.utils.data
import torch.nn as nn
from absl import app


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

    train(dataset, num_epochs=10)


def train(dataset, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=3, shuffle=True)
    loss = nn.CrossEntropyLoss()
    model = LSTM(FLAGS.num_ceps, dataset.num_labels, size_hidden_layers=100)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    for epoch in tqdm(range(num_epochs), desc='training epochs'):

        for batch in tqdm(train_loader, desc="training batches"):
            sample, target = batch
            sample.to(device)
            target.to(device)
            optimizer.zero_grad()
            prediction = model.forward(sample)
            #print(prediction.shape)
            #print(target.shape)

            loss_val = loss(prediction.squeeze(), target.squeeze())
            #print("loss", loss_val)
            loss_val.backward()
            optimizer.step()
        print(loss_val)


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