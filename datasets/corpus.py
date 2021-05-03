import os.path

import numpy as np
import scipy.io.wavfile

class Corpus:
    """Corpus object for the TIMIT corpus
    Largely adapted from: https://github.com/rash-me-not/TIMIT-phoneme-recognition-with-Recurrent-Neural-Nets 
    """
    
    audioext = ".WAV"
    
    # List of phoneme symbols used in phone-level transcriptions 
    phones = ["b", "d", "g", "p", "t", "k", "dx", "q", 
              "jh", "ch", "s", "sh", "z", "zh", "f", "th", 
              "v", "dh", "m", "n", "ng", "em", "en", "eng", 
              "nx", "l", "r", "w", "y", "hh", "hv", "el", 
              "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", 
              "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", 
              "ax", "ix", "axr", "ax-h", "pau", "epi", 
              # closure portion of stops
              "bcl", "dcl", "gcl", "pcl", "tcl", "kcl",
              # silence
              "h#"]
    
    silence = "h#"

    # Create dictionary to map phonemes to numbers, e.g. phone2int['g'] = 2
    phone2int = dict([(key,val) for val,key in enumerate(phones)])

    def __init__(self, transcription_dir, sample_file_dir):
        """Corpus(transcription_dir, audio_dir)
        Create a corpus object with transcriptions and audio files
        in the specified create_directories
        """
        self.transcriptions = os.path.realpath(transcription_dir)
        self.feature_extractor = None

        # Find sample rate, traverse audio directory for first wav file we find
        self.Fs = scipy.io.wavfile.read(transcription_dir + sample_file_dir + ".WAV")[0]

        # Walk iterates over lists of subdirectories and files in
        # a directory tree. 
        #for directory, _subdirs, files in os.walk(transcription_dir + sample_file_dir):
        #    for f in files:
        #        if f.endswith(self.audioext.upper()):
        #            print(directory)
                    # Found one, get sample rate and stop looking
        #            self.Fs, _ = \
        #                scipy.io.wavfile.read(os.path.join(directory, f))
        #            break 
        print(self.Fs)

    @classmethod
    def get_phonemes(cls):
        """"get_phonemes() 
        Return phone names used in transcriptions.
        See TIMIT doc/phoncode.doc for details on meanings
        """
        return cls.phones

    @classmethod
    def get_phones_ID(cls, phoneme):
        """"get_phones_ID() 
        Return phone IDs after mapping phonemes to numbers
        """
        return cls.phone2int[phoneme]

    def get_phone_transcription(self, utterance, Fs=None):
        """get_phone_transcription(utterance)
        Given a relative path to a specific utterance, e.g.
            train/dr1/jcjf0/sa1  (training directory, dialect region
                1, speaker jcjf0, sentence sa1 - see TIMIT docs)
        return the phonemes and samples/timing with which they are associated.
        
        If Fs is None, returns sample numbers from TIMIT documentation,
        otherwise converts to seconds as per Fs
        
        Returns:
        starts, stops, phones
        """
        
        # Construct location of phoneme transcription file
        # phnfile = os.path.join(self.transcriptions, utterance + ".phn")
        phnfile = self.transcriptions + utterance + ".PHN"
        
        starts = []
        stops = []
        phones = []
        
        # Loop through each line pulling out start and stop samples
        # as well as the label and put into a l
        for line in open(phnfile, 'r'):
            b,e,p = line.split()
            starts.append(int(b))
            stops.append(int(e))
            phones.append(p)
        
        # Cast to numpy array
        starts = np.asarray(starts)
        stops = np.asarray(stops)
        # Convert to seconds if Fs provided
        if Fs is not None:
            starts = starts / Fs
            stops = stops / Fs            
            
        return starts, stops, phones
    
    def get_audiofilename(self, audio_dir, utterance):
        """get_audiofilename(utterance)
        Given a relative path to a specific utterance, e.g. 
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        
        audfile = os.path.join(audio_dir, utterance + ".wav")
        return audfile
    
    def get_relative_audiofilename(self, utterance):
        """get_relative_audiofilename(utterance)
        Given a relative path to a specific utterance, e.g. 
            train/dr1/jcjf0/sa1
        construct a relative pathname from the root of the corpus
        audio data to the audio file associated with it:
            train/dr1/jcjf0/sa1.wav
        """
        return utterance + ".wav"
    
    def get_features(self, utterance):
        """get_features(utterance)
        Return features associated with the utterance.
        User must have called set_feature_extractor and provided a class
        with a method get_features(filename)
        """
        if self.feature_extractor:
            return self.feature_extractor.get_features(
                self.get_relative_audiofilename(utterance))
        else:
            raise RuntimeError(
                'Cannot extract features until set_feature_extractor is called')

    def get_labels(self, utterance):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames
        
        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]
        
        """
        
        if self.feature_extractor:
            # Get phone alignment in seconds
            start, stop, phones = self.get_phone_transcription(
                utterance, self.Fs)
            
            # Features advance every N seconds
            adv_s = self.feature_extractor.get_advance_s()
            
            # Align the labels with the frame rate
            start = np.around(start / adv_s)
            stop[0:-1] = start[1:]-1  # Drive stop frames by next starts
            stop[-1] = np.around(stop[-1] / adv_s)  # Last stop
            
            return start, stop, phones
            
        else:
            raise RuntimeError(
                'Cannot get_labels until set_feature_extractor is called')      
    
    def get_silence(self):
        "get_silence() - Return label for silence/noise"
        return self.silence    
    
    def get_utterances(self, audio_dir, utttype="train"):
        """get_utterances(utttype)
        Return list of train or test utterances (as specified by utttype)
        
        e.g.  get_utterances('train')
        returns a list:
            [train/dr1/jcjf0/sa1, train/dr1/jcjf0/sa2, ...
             train/dr8/mtcs0/sx352]
        """
        
        utterances = []  # list of utterances
        
        # Traverse audio directory
        targetdir = os.path.join(audio_dir, utttype)
        
        # Walk iterates over lists of subdirectories and files in
        # a directory tree. 
        for directory, _subdirs, files in os.walk(targetdir):
            # Get rid of root and remove file separator
            reldir = directory.replace(audio_dir, "")[1:]
            
            for f in files:
                if f.lower().endswith(".wav"):
                    # Found one, strip off extension and add to list
                    uttid, _ext = os.path.splitext(f)
                    utterances.append(os.path.join(reldir, uttid))
        
        # Return the list as a numpy array which will let us use
        # more sophisticated indexing (e.g. using a list variable
        # of indices
        return np.asarray(utterances)
   
    def set_feature_extractor(self, feature_extractor):
        """set_feature_extractor(feature_extractor)
        After passing in an object capable of extracting features with
        a call to get_features(fname) with a filename relative to the
        audio root, one can use get_features in this class to retrieve
        features
        """
        if not hasattr(feature_extractor, 'get_features'):
            raise RuntimeError(
                'Specified feature_extractor does not support get_features')
            
        self.feature_extractor = feature_extractor
    
    def categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]
    
    @classmethod
    def phones_to_onehot(cls, phones):
        """"phones_to_onehot(phones)
        Given a list of phones, convert them to one hot vectors
        """
        N = len(cls.phones)
        phonenum = [cls.phone2int[p] for p in phones]
        return cls.categorical(phonenum, N)