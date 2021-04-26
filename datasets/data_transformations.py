import torchaudio


class MFCC(object):
    """Extract MFCC coefficients

    Args:
    """

    def __init__(self):
        print("TEST")

    def __call__(self, sample):
        audio, phonemes = sample['audio'], sample['phonemes']

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40)

        audio = mfcc_transform(audio)

        return {'audio': audio, 'phonemes': phonemes}
