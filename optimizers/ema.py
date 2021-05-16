import torch


class ExponetialMovingAverage(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha

    def step(self):
        for param, ema_param in zip(list(self.model.state_dict().values()), list(self.ema_model.state_dict().values())):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1.0 - self.alpha))
