import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, mfccs, output_phonemes, units_per_layer, num_layers, dropout, name='vanillaLSTMfullylabeled.pth'):
        super(LSTM, self).__init__()
        '''input: number of mfcc coefficients, number of output phonemes, layer dimensions in a list
                '''
        self.name = name
        self.hidden_layers = nn.LSTM(mfccs, units_per_layer, num_layers=num_layers, dropout=dropout, batch_first= False)
        self.output_layer = nn.Linear(units_per_layer, output_phonemes)

    def forward(self, x):
        x, _ = self.hidden_layers(x)
        return self.output_layer(x)

