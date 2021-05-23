import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optimizers import ema
from optimizers.ema import ExponetialMovingAverage
from models.lstm1 import LSTM
from torch.autograd import Variable

from optimizers.AdaNormGrad import AdamNormGrad

class MeanTeacher(nn.Module):

    # Problems?
    # 1. Random aug. or noise
    # 2. No softmax is used
    # 3. Batch size = 1. In the paper they have for example 40 unlabeled samples and 20 labeled samples in the batch

    # Mean Teacher algorithm:
    # 1. Take a supervised architecture and make a copy of it. Let's call the original model the student and the new one the teacher.
    # 2. At each training step, use the same minibatch as inputs to both the student and the teacher but add random augmentation or noise to the inputs separately.
    # 3. Add an additional consistency cost between the student and teacher outputs (after softmax).
    # 4. Let the optimizer update the student weights normally.
    # 5. Let the teacher weights be an exponential moving average (EMA) of the student weights. That is, after each training step, update the teacher weights a little bit toward the student weights.

    def __init__(self, mfccs, output_phonemes, units_per_layer, num_layers, dropout, optimizer, lr, max_steps=10000, ema_decay=0.999, consistency_weight=1.0):
        super(MeanTeacher, self).__init__()

        self.name = 'MeanTeacher'
        self.consistency_weight = consistency_weight
        self.current_consistency_weight = 0
        
        self.max_steps = max_steps
        self.step = 0

        self.std = 0.15  # Need to check
        self.mean = 0.0

        self.loss_consistency = nn.MSELoss()
        self.loss_class = nn.CrossEntropyLoss()

        self.student = self.model = LSTM(mfccs, output_phonemes, units_per_layer,
                          num_layers, dropout, name='student.pth')
        self.teacher = self.model = LSTM(mfccs, output_phonemes, units_per_layer,
                          num_layers, dropout, name='teacher.pth')

        self.teacher.load_state_dict(self.student.state_dict())

        self.ema_optimizer = ExponetialMovingAverage(
            model=self.student, ema_model=self.teacher, alpha=ema_decay)

        if (optimizer == 'Adam'):
            # Configuring the Optimizer (ADAptive Moments)
            self.optimizer = torch.optim.Adam(
                self.student.parameters(), lr=lr)
        else:
            # Configuring the Optimizer (ADAptive Moments but with normalizing gradients)
            self.optimizer = AdamNormGrad(self.student.parameters(), lr=lr)


    def to(self, device):
        self.student = self.student.to(device)
        self.teacher = self.teacher.to(device)

    def get_optimizer(self):
        return self.optimizer

    def forward_student(self, x):
        return torch.squeeze(self.student(x), dim=1)

    def forward_teacher(self, x):
        return torch.squeeze(self.teacher(x), dim=1)

    def forward(self, x):
        return self.forward_student(x)

    def loss_fn(self, device, sample, targets):
        loss = 0

        sample = sample.to(device)

        if not(targets is None):
            targets = targets.to(device)
            loss += self.loss_class(self.forward_student(sample +
                                    torch.randn(sample.size()).to(device) * self.std), targets)
            #print('loss after cross entropy', loss)

        loss += self.current_consistency_weight * self.loss_consistency(self.forward_student(sample + torch.randn(sample.size()).to(device) * self.std),
                                                                self.forward_teacher(sample + torch.randn(sample.size()).to(device) * self.std))
        #print('loss after consistency', loss)

        return loss

    def train_step(self, loss_val):
        loss_val.backward()
        self.optimizer.step()
        self.ema_optimizer.step()

    def linear_ramp_up(self):
        return min(float(self.step) / self.max_steps, 1.0)

    def update_rampup(self, epoch, rampup_length):
        self.current_consistency_weight = self.consistency_weight * self.sigmoid_rampup(epoch, rampup_length)

    def sigmoid_rampup(self, current, rampup_length):
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))