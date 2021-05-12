# import torch.cuda
# import torch.nn as nn

# class LSTM(nn.Module):

#     def __init__(self, mfccs, output_phonemes, size_hidden_layers):
#         super(LSTM, self).__init__()
#         '''input: number of mfcc coefficients, number of output phonemes, layer dimensions in a list
#                 '''

#         self.hidden_layers = nn.LSTM(mfccs, size_hidden_layers, num_layers=5)
#         '''self.output_layer = nn.Sequential(nn.Linear(size_hidden_layers, output_phonemes),
#                             nn.Softmax())'''
#         self.output_layer = nn.Linear(size_hidden_layers, output_phonemes)

#     def forward(self, x):
#         x, _ = self.hidden_layers(x)
#         return self.output_layer(x)

import torch
import torch.nn as nn

torch.manual_seed(1)

# parameters of LSTM
input_dim = 13 
hidden_dim = 500   
layer_dim = 1
output_dim = 48 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        x = x.to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
    
##### HOW TO RUN LSTM MODEL
# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)


# # checking if dimensions are correct
# for i in range(len(list(model.parameters()))):
#     print('HERE')
#     print(list(model.parameters())[i].size())
