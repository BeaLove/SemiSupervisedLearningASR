from absl import app
from absl import flags
from pathlib import Path
import pandas as pd
import os
from scipy.io import wavfile
from sphfile import SPHFile
import scipy.io

FLAGS = flags.FLAGS

def main(argv):
  del argv  # Unused.
  print("Hello these are your settings: ")
  print(FLAGS.winlen)
  print(FLAGS.winstep)

  importFileInfo('test_data.csv', '../timit')

def importFileInfo(filename, rootDir):
    df = pd.read_csv(os.path.join(rootDir, filename))
    print(df.head())

    audio_paths = df[df['is_audio'] == True]['path_from_data_dir']

    for path in list(audio_paths):
        print(os.path.abspath(path))
        #with path.open()
        wavfile.read(os.path.normpath(os.path.join(rootDir, 'data', path)))
        break

if __name__ == '__main__':
    flags.DEFINE_float('winlen', 0.02, 'windowing length')
    flags.DEFINE_float('winstep', 0.01, 'Windowing step')
    flags.DEFINE_integer('numcep', 13, 'Number of MFCC coefficients')
    flags.DEFINE_integer('nfilt', 26, 'Number of filters')
    flags.DEFINE_float('preemph', 0.97, 'Preemphasis filter coefficient')
    app.run(main)