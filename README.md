# SemiSupervisedLearningASR

Experiments on semi-supervised learning for automatic speech recognition.

Architectures:
- LSTMs

Dataset:
- TIMIT
- 
### Install dependencies

```bash
python3 -m venv venv
. venv/bin/activate
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
