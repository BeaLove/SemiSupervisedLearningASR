import torch

class ExponetialMovingAverage(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha

        for ema_param in self.ema_model.parameters()):
            ema_param.detach_()

    def step(self):

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.alpha).add_((1 - self.alpha) * param)

            #w = w * 0.99 + 0.01 * new_w
