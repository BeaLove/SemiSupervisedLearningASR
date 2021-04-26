import torch
import numpy as np
import transform

class preProcessing:

    def __init__(self, config):
        self.winlength = config.winlength
        self.winstep = config.winstep
        self.numcep = config.numcep
        self.preemph = config.preemph
        self.numfilt = config.numfilt

    def createMFCC(self, data):