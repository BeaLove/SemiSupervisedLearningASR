import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from optimizers.ema import ExponetialMovingAverage
from models.lstm1 import LSTM
from torch.autograd import Variable


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

    def __init__(self, mfccs, output_phonemes, size_hidden_layers, max_steps=10000, ema_decay=0.999, consistency_weight=10.0):
        super(MeanTeacher, self).__init__()

        self.name = 'MeanTeacher'
        self.consistency_weight = consistency_weight
        self.max_steps = max_steps
        self.step = 0

        self.std = 10.0
        self.mean = 0.0

        self.student = LSTM(mfccs, output_phonemes, size_hidden_layers)
        self.teacher = LSTM(mfccs, output_phonemes, size_hidden_layers)

        self.teacher.load_state_dict(self.student.state_dict())


        self.ema_optimizer = ExponetialMovingAverage(
            model=self.student, ema_model=self.teacher, alpha=ema_decay)

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)

    def to(self, device):
        self.student = self.student.to(device)
        self.teacher = self.teacher.to(device)

    def get_optimizer(self):
        return self.optimizer

    def consistency_criterion(self, teacher_outputs, student_outputs):
        loss = nn.MSELoss()
        return loss(teacher_outputs, student_outputs)

    def criterion(self, outputs, labels):
        loss = nn.CrossEntropyLoss()
        return loss(outputs, labels)

    def forward_student(self, x):
        return torch.squeeze(self.student(x), dim=0)

    def forward_teacher(self, x):
        return torch.squeeze(self.teacher(x), dim=0)

    def forward(self, x):
        return self.forward_student(x)

    def loss_fn(self, outputs, labels):
        return self.criterion(outputs, labels)

    def train_step(self, device, u_data, l_data, target):
        self.optimizer.zero_grad()

        u_data = u_data + \
            torch.randn(u_data.size()).to(device) * self.std + self.mean
        l_data = l_data + \
            torch.randn(l_data.size()).to(device) * self.std + self.mean

        target = torch.squeeze(target, dim=0)
        loss_class = self.criterion(self.forward_student(l_data), target)

        loss_consistency = self.consistency_criterion(
            self.forward_student(u_data), self.forward_teacher(u_data))

        loss = loss_class + self.consistency_weight * \
            loss_consistency * self.linear_ramp_up()

        loss.backward()

        self.optimizer.step()
        self.ema_optimizer.step()

        self.step += 1

        return loss

    def linear_ramp_up(self):
        return min(float(self.step) / self.max_steps, 1.0)
