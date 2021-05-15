import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from optimizers.ema import WeightEMA
from models.lstm1 import LSTM
from torch.autograd import Variable


class Baseline(nn.Module):

    # Todo
    # Loss
    # Dropout or add noise - read how

    # Mean Teacher algorithm:
    # 1. Take a supervised architecture and make a copy of it. Let's call the original model the student and the new one the teacher.
    # 2. At each training step, use the same minibatch as inputs to both the student and the teacher but add random augmentation or noise to the inputs separately.
    # 3. Add an additional consistency cost between the student and teacher outputs (after softmax).
    # 4. Let the optimizer update the student weights normally.
    # 5. Let the teacher weights be an exponential moving average (EMA) of the student weights. That is, after each training step, update the teacher weights a little bit toward the student weights.

    def __init__(self, mfccs, output_phonemes, size_hidden_layers):
        super(Baseline, self).__init__()

        self.name = 'Baseline'

        self.model = LSTM(mfccs, output_phonemes, size_hidden_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    def to(self, device):
        self.model = self.model.to(device)

    def criterion(self, outputs, labels):
        loss = nn.CrossEntropyLoss()
        return loss(outputs, labels)

    def get_optimizer(self):
        return self.optimizer

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=0)

    def loss_fn(self, outputs, labels):
        return self.criterion(outputs, labels)

    def train_step(self, device, u_data, l_data, target):
        self.optimizer.zero_grad()

        target = torch.squeeze(target, dim=0)
        loss = self.criterion(self.forward(l_data), target)

        loss.backward()

        self.optimizer.step()

        return loss
