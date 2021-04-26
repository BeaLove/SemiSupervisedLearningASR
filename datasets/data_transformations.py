import torchaudio
import torch

class MFCC(object):
    """Extract MFCC coefficients

    Args:
    """

    def __init__(self, n_mfcc, preemph):
        self.n_mfcc = n_mfcc
        self.preemph = preemph

    def __call__(self, sample):
        print("TRANSFORM")
        audio, phonemes = sample['audio'], sample['phonemes']

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=self.n_mfcc)

        audio = torchaudio.functional.lfilter(audio, torch.tensor([1.0, 0.0]), torch.tensor([1.0, -self.preemph]))

        audio = mfcc_transform(audio)

        return {'audio': audio, 'phonemes': phonemes}
