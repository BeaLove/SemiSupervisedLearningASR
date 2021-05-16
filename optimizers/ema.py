import torch


class ExponetialMovingAverage(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha

    def step(self):

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.alpha).add_(param.data, 1 - self.alpha)