import torchaudio


class MFCC(object):
    """Extract MFCC coefficients

    Args:
    """

    def __init__(self):

    def __call__(self, sample):
        audio, phonemes = sample['audio'], sample['phonemes']

        audio = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, dct_type=2, norm='ortho')

        return {'audio': audio, 'phonemes': phonemes}
