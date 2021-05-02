import os.path

import numpy as np

class Phonemes:
    phonemes_train = []
    phonemes_test = []
    
    def __init__(self, corpus, parent_dir):
        """ Create a phonemes object for a given corpus.
        """
        self.corpus = corpus
        self.parent_dir = parent_dir
        
        for path, subdirs, files in os.walk(parent_dir + "/data/TRAIN"):
            for name in files:
                curr = os.path.join(path, name)
                
                # Detected a phoneme file: loading phonemes for one audio file
                if(".PHN" in name):
                
                    # Get start, end, transcription of all the phonemes in the audio file
                    phones_start, phones_stop, phonemes = corpus.get_phone_transcription(curr)
                    
                    # Indices (numerical IDs) of the phonemes
                    phonemes_ids = []
                    for phoneme in phonemes:
                        phonemes_ids.append(corpus.get_phones_ID(phoneme))
                        
                    # One-hot representation of the indices of the phonemes
                    phonemes_ids_1hot = corpus.phones_to_onehot(phonemes)
                    
                    self.phonemes_train.append([curr, phones_start, phones_stop, phonemes, phonemes_ids, phonemes_ids_1hot])
            
            for path, subdirs, files in os.walk(parent_dir + "/data/TEST"):
                for name in files:
                    curr = os.path.join(path, name)
                    
                    # Detected a phoneme file: loading phonemes for one audio file
                    if(".PHN" in name):
                    
                        # Get start, end, transcription of all the phonemes in the audio file
                        phones_start, phones_stop, phonemes = corpus.get_phone_transcription(curr)
                        
                        # Indices (numerical IDs) of the phonemes
                        phonemes_ids = []
                        for phoneme in phonemes:
                            phonemes_ids.append(corpus.get_phones_ID(phoneme))
                            
                        # One-hot representation of the indices of the phonemes
                        phonemes_ids_1hot = corpus.phones_to_onehot(phonemes)
                        
                        self.phonemes_test.append([curr, phones_start, phones_stop, phonemes, phonemes_ids, phonemes_ids_1hot])
    
    def getTrainElements(self):
        """ Getter for elements in phonemes of the train set. """
        return self.phonemes_train
        
    def getTestElements(self):
        """ Getter for elements in phonemes of the test set """
        return self.phonemes_test