import torch.cuda
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, mfccs, output_phonemes, size_hidden_layers):
        super(LSTM, self).__init__(self)
        '''input: number of mfcc coefficients, number of output phonemes, layer dimensions in a list
                '''
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        torch.cuda.device(dev)
        self.hidden_layers = nn.LSTM(mfccs, size_hidden_layers, num_layers=5)
        self.output_layer = nn.Sequential(nn.Linear(size_hidden_layers, output_phonemes),
                            nn.Softmax())

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

