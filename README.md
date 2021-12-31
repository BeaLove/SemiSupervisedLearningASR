# SemiSupervisedLearningASR

Experiments on semi-supervised learning for automatic speech recognition. Utilizes the Mean Teacher model ("Mean Teachers are Better Role Models", Tarvainen & Valpola, 2017) as well as the "vanilla" soft-target semi-supervised learning method.

Architectures:
- LSTMs

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

### Test plotting MFCC coefficients

This can be used to plot the result after running MFCC for the TIMIT dataset.
```bash
python3 app_test.py --n_fft=512
```

### Train

This can be used to train a model by using an experiment config. For example:
```bash
python3 train.py --flagfile=experiments/<experiment config>
```

If you want you can specify the flags by using command line args. For example:
```bash
python3 train.py --results_save_dir=results --dataset_root_dir='timit' --num_epochs=100
```
