import torchaudio
import torch
import math
import copy


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

        frame_length = self.n_fft / sample_rate * math.pow(10, 3)
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
            "num_ceps": self.num_ceps,
            "snip_edges": False
        }

        audio = torch.tensor(audio, dtype=torch.float) 
        audio = torchaudio.compliance.kaldi.mfcc(audio, **params)

       # print("AU1", audio.shape[0])

        return {'audio': audio, 'phonemes': phonemes, 'max_frames': audio.shape[0]}


class Phonemes(object):

    def __init__(self, n_fft, preemphasis_coefficient, num_ceps, corpus):
        self.n_fft = n_fft
        self.preemphasis_coefficient = preemphasis_coefficient
        
        self.num_ceps = num_ceps
        self.corpus = corpus
        
    def samples_to_ms(self, samples):
        ms_equivalent = (samples / self.corpus.Fs) * math.pow(10, 3)
        return ms_equivalent
    
    def __call__(self, sample):
        audio, phonemes, max_frames = sample['audio'], sample['phonemes'], sample['max_frames']
        frames_start = []
        frames_end = []

        frame_length = self.n_fft / self.corpus.Fs * math.pow(10, 3)    # maybe not?
        frame_shift = frame_length / 2.0

        start_time = 0
        end_time = self.samples_to_ms(phonemes[len(phonemes)-1][1])
        current_time = start_time

        while(current_time < end_time and len(frames_start) < max_frames):
            frames_start.append(current_time)
            frames_end.append(current_time + frame_length)
            current_time += frame_shift
        frames_end[len(frames_end) - 1] = end_time

       # print("FRA", len(frames_end))

        phonemes_per_frame = []

        for i in range(0, len(frames_start)):
            phonemes_current_frame = []
            frame_start = frames_start[i]
            frame_end = frames_end[i]

            for j in range(0, len(phonemes)):
                phoneme_start = self.samples_to_ms(phonemes[j][0])
                phoneme_end = self.samples_to_ms(phonemes[j][1])
                phoneme = phonemes[j]

                if (phoneme_start >= frame_start and phoneme_start < frame_end):
                    # the phoneme starts in this frame
                    phonemes_current_frame.append(phoneme)
                elif (phoneme_end >= frame_end and phoneme_end < frame_end):
                    # the phoneme end in this frame
                    phonemes_current_frame.append(phoneme)

            phonemes_per_frame.append(phonemes_current_frame)

        phonemes_first_frame = phonemes_per_frame[0]
        if(len(phonemes_first_frame) == 0):
            # Patching silence in the first frame if it's initially empty
            phonemes_per_frame[0].append([0, 0, 'h#'])

        for i in range(0, len(frames_start)):
            phonemes_current_frame = phonemes_per_frame[i]
            
            if(len(phonemes_current_frame) == 0 and i > 0):
                # the phoneme starts in a previous and ends in a next frame
                phonemes_per_frame[i] = copy.deepcopy(phonemes_per_frame[i-1])
                phonemes_per_frame[i] = [phonemes_per_frame[i][len(phonemes_per_frame[i]) - 1][0:3]]
                
        for i in range(0, len(frames_start)):
            phonemes_current_frame = phonemes_per_frame[i]
            if(len(phonemes_current_frame) == 2):
                # we have exactly 2 phonemes uttered in the frame
                phoneme1 = phonemes_per_frame[i][0]
                phoneme2 = phonemes_per_frame[i][1]
                
                frame_start = frames_start[i]
                frame_end = frames_end[i]
                frame_duration = frames_end[i] - frames_start[i]

                phoneme1_end = self.samples_to_ms(phoneme1[1])
                phoneme2_start = self.samples_to_ms(phoneme2[0])
                percentage_phoneme1 = (phoneme1_end - frame_start)/frame_duration
                percentage_phoneme2 = (frame_end - phoneme2_start)/frame_duration
                
                phonemes_per_frame[i][0] = copy.deepcopy(phonemes_per_frame[i][0])
                phonemes_per_frame[i][1] = copy.deepcopy(phonemes_per_frame[i][1])
                
                phonemes_per_frame[i][0].append(percentage_phoneme1)
                phonemes_per_frame[i][1].append(percentage_phoneme2)

            elif(len(phonemes_current_frame) > 2):
                # we have 3+ phonemes uttered in the frame which might never happen though
                phonemes_in_frame = len(phonemes_current_frame)
                phoneme_first = phonemes_per_frame[i][0]
                phoneme_last = phonemes_per_frame[i][phonemes_in_frame - 1]
                
                frame_start = frames_start[i]
                frame_end = frames_end[i]
                frame_duration = frames_end[i] - frames_start[i]

                phoneme_first_end = self.samples_to_ms(phoneme_first[1])
                phoneme_last_start = self.samples_to_ms(phoneme_last[0])
                percentage_phoneme_first = (phoneme_first_end - frame_start)/frame_duration
                percentage_phoneme_last = (frame_end - phoneme_last_start)/frame_duration
                
                phonemes_per_frame[i][0] = copy.deepcopy(phonemes_per_frame[i][0])
                phonemes_per_frame[i][phonemes_in_frame - 1] = copy.deepcopy(phonemes_per_frame[i][phonemes_in_frame - 1])
                
                phonemes_per_frame[i][0].append(percentage_phoneme_first)
                phonemes_per_frame[i][phonemes_in_frame - 1].append(percentage_phoneme_last)

                for j in range(1, phonemes_in_frame - 1):
                    phoneme_middle = phonemes_per_frame[i][j]
                    phoneme_middle_start = self.samples_to_ms(phoneme_middle[0])
                    phoneme_middle_end = self.samples_to_ms(phoneme_middle[1])
                    percentage_phoneme_middle = (phoneme_middle_end - phoneme_middle_start)/frame_duration
                    
                    phonemes_per_frame[i][j] = copy.deepcopy(phonemes_per_frame[i][j])
                    
                    phonemes_per_frame[i][j].append(percentage_phoneme_middle)

        return phonemes_per_frame, len(frames_start)
        
