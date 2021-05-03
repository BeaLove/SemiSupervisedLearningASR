import torch.utils.data
import torch.nn
from absl import app


from tqdm import tqdm

from models.lstm import *

from absl import flags
from datasets.TIMITdataset import TimitDataset

FLAGS = flags.FLAGS

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





dataset = TimitDataset(csv_file=FLAGS.csv_file, root_dir=FLAGS.root_dir,
                         pre_epmh=FLAGS.preemphasis_coefficient,
                         num_ceps=FLAGS.num_ceps, n_fft=FLAGS.n_fft, frame_size=FLAGS.frame_len, frame_shift=FLAGS.frame_shift)


def main(args):

    train(dataset, num_epochs=FLAGS.epochs)


def train(dataset, num_epochs = FLAGS.epochs):


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=3, shuffle=True)
    model = LSTM(FLAGS.num_ceps, dataset.num_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs), desc='training epochs'):

        for batch in tqdm(train_loader, desc="training batches"):
            #TODO call forward, loss_func, optimizer etc in right order haha
            optimizer.zero_grad()
            model.forward(batch)



if __name__ == '__main__':
    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')

    app.run(main)