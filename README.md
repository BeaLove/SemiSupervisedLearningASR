# SemiSupervisedLearningASR

Experiments on semi-supervised learning for automatic speech recognition.

Architectures:
- LSTMs (maybe bi-directional)

Dataset:
- TIMIT


### Activate virtual environment

```bash
python3 -m venv venv
. venv/bin/activate
```

#### Using Anaconda in MacOS/Linux

```bash
conda create --name venv python=3.8 numpy
source activate venv
```


#### Using Anaconda in Windows

```bash
conda create --name venv python=3.8 numpy scipy
activate venv
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install dataset
```bash
wget https://data.deepai.org/timit.zip
unzip timit.zip
rm timit.zip
```

## Running

### Test

This can be used to plot the result after running MFCC for the TIMIT dataset.
```bash
python3 app_test.py --n_fft=512
```
