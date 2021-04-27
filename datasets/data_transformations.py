import torchaudio
import torch


class MFCC(object):
    """Extract MFCC coefficients
    Args:
    """

    def __init__(self, n_fft, preemphasis_coefficient, num_ceps):
        self.n_fft = n_fft
        self.preemphasis_coefficient = preemphasis_coefficient
        self.num_ceps = num_ceps

    def __call__(self, sample, sample_rate):
        audio, phonemes = sample['audio'], sample['phonemes']

        frame_length = self.n_fft / sample_rate * 1000.0
        frame_shift = frame_length / 2.0

        params = {
            "channel": 0,
            "dither": 0.0,
            "window_type": "hamming",
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "remove_dc_offset": False,
            "round_to_power_of_two": False,
            "sample_frequency": sample_rate,
            "preemphasis_coefficient": self.preemphasis_coefficient,
            "num_ceps": self.num_ceps
        }

        audio = torch.tensor(audio, dtype=torch.float)
        audio = torchaudio.compliance.kaldi.mfcc(audio, **params)

        return {'audio': audio, 'phonemes': phonemes}
