from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio

import ctypes
import multiprocessing as mp

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TimitDataset(Dataset):
    """Timit dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.audio_frame = self.audio_frame[self.audio_frame['is_phonetic_file'] == True]

        self.root_dir = root_dir
        self.transform = transform

        self.use_cache = False
        self.cache = {}

    def __len__(self):
        return len(self.audio_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.root_dir, 'data',
                            self.audio_frame.iloc[idx]['path_from_data_dir'])
        path = os.path.splitext(path)[0]

        audio_name = path + '.WAV.wav'
        audio, sample_rate = torchaudio.load(audio_name)

        phonemes_name = path + '.PHN'

        with open(phonemes_name, 'r') as phonemes_file:
            phonemes = [l.strip().split(' ') for l in phonemes_file]
            phonemes = [[int(hd), int(tl), w] for (hd, tl, w) in phonemes]

        phonemes = list(phonemes)

        sample = {'audio': audio, 'phonemes':  phonemes}

        if self.transform:
            sample = self.transform(sample, sample_rate)

        return sample

import torch
import torchvision
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from datasets.data_transformations import train_transformations, validation_transformations
from torch.utils.data import Subset

class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, num_transformations, num_classes, indexes=None, split='train', validation=False,
                 transform=None, download=False):
        super(SVHN, self).__init__(
            root,
            split=split,
            transform=transform,
            download=download
        )

        self.validation = validation
        self.num_classes = num_classes
        self.num_transformations = num_transformations

        if indexes is not None:
            self.data = self.data[indexes]
            self.labels = np.array(self.labels)[indexes]

        self.data = [Image.fromarray(np.transpose(image, (1, 2, 0))) for image in self.data]

    def __getitem__(self, index):
        image, target = self.data[index], self.labels[index]

        # this is the validation case where no transformation is applied
        if self.validation:
            return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), target

        return self.transform_image(image), target

    def transform_image(self, image):
        return [self.transform(image) for _ in range(self.num_transformations)]


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, num_transformations, num_classes, indexes=None, train=True, validation=False,
                 transform=None, download=False):
        super(CIFAR10, self).__init__(
            root,
            train=train,
            transform=transform,
            download=download
        )

        self.validation = validation
        self.num_classes = num_classes
        self.num_transformations = num_transformations

        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]

        self.data = [Image.fromarray(image) for image in self.data]

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]

        # this is the validation case where no transformation is applied
        if self.validation:
            return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), target

        return self.transform_image(image), target

    def transform_image(self, image):
        return [self.transform(image) for _ in range(self.num_transformations)]


def train_val_split(labels, labels_per_class):
    labels = np.array(labels)
    train_labeled_indexes = []
    train_unlabeled_indexes = []
    validation_indexes = []

    for i in range(10):
        indexes = np.where(labels == i)[0]
        np.random.shuffle(indexes)

        train_labeled_indexes.extend(indexes[:labels_per_class])
        train_unlabeled_indexes.extend(indexes[labels_per_class:-500])
        validation_indexes.extend(indexes[-500:])

    np.random.shuffle(train_labeled_indexes)
    np.random.shuffle(train_unlabeled_indexes)
    np.random.shuffle(validation_indexes)

    return train_labeled_indexes, train_unlabeled_indexes, validation_indexes


def load_cifar10_data(config):
    transformations_train = train_transformations()
    transformations_validation = validation_transformations()

    dataset = torchvision.datasets.CIFAR10(
        root=config.root,
        train=True,
        download=True,
        transform=transformations_train
    )


    if config.labeled_data == 'all':
        config.labeled_data = len(dataset)

    labeled_data_per_class = int(config.labeled_data / len(dataset.classes))
    train_labeled_indexes, train_unlabeled_indexes, validation_indexes = train_val_split(dataset.targets,
                                                                                         labeled_data_per_class)

    train_labeled_dataset = CIFAR10(
        config.root,
        1,
        config.dataset_classes,
        train_labeled_indexes,
        transform=transformations_train,
        train=True
    )

    train_unlabeled_dataset = CIFAR10(
        config.root,
        config.k,
        config.dataset_classes,
        train_unlabeled_indexes,
        transform=transformations_train,
        train=True
    )

    validation_dataset = CIFAR10(
        config.root,
        config.k,
        config.dataset_classes,
        validation_indexes,
        train=True,
        validation=True,
        transform=transformations_validation
    )

    if not train_unlabeled_indexes:
        train_unlabeled_dataset = None

    return train_labeled_dataset, train_unlabeled_dataset, validation_dataset


def load_svhn_data(config):
    transform_train = train_transformations()
    transformations_validation = validation_transformations()

    dataset = torchvision.datasets.SVHN(
        config.root,
        split='train',
        transform=None,
        target_transform=None,
        download=True
    )

    if config.labeled_data == 'all':
        config.labeled_data = len(dataset)

    labeled_data_per_class = int(config.labeled_data / 10)
    train_labeled_indexes, train_unlabeled_indexes, validation_indexes = train_val_split(dataset.labels,
                                                                                         labeled_data_per_class)

    train_labeled_dataset = SVHN(
        config.root,
        1,
        config.dataset_classes,
        train_labeled_indexes,
        transform=transform_train,
        split='train'
    )

    train_unlabeled_dataset = SVHN(
        config.root,
        config.k,
        config.dataset_classes,
        train_unlabeled_indexes,
        transform=transform_train,
        split='train'
    )

    validation_dataset = SVHN(
        config.root,
        config.k,
        config.dataset_classes,
        validation_indexes,
        split='train',
        validation=True,
        transform=transformations_validation
    )

    if not train_unlabeled_indexes:
        train_unlabeled_dataset = None

    return train_labeled_dataset, train_unlabeled_dataset, validation_dataset


def load_test_data(arguments):
    test_transformations = validation_transformations()
    if arguments.dataset_name == 'CIFAR10':
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=test_transformations
        )
    else:
        test_dataset = torchvision.datasets.SVHN(
            root='./data',
            split='test',
            download=True,
            transform=test_transformations
        )

    return DataLoader(test_dataset, batch_size=arguments.batch_size)


def load_train_data(config):
    if config.dataset_name == 'CIFAR10':
        train_labeled_dataset, train_unlabeled_dataset, validation_dataset = load_cifar10_data(config)
    elif config.dataset_name == 'SVHN':
        train_labeled_dataset, train_unlabeled_dataset, validation_dataset = load_svhn_data(config)
    else:
        raise Exception(f'Dataset {config.dataset_name} is not supported')

    train_labeled_dataloader = DataLoader(
        train_labeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True
    )

    if not train_unlabeled_dataset:
        return train_labeled_dataloader, None, validation_dataloader

    train_unlabeled_dataloader = DataLoader(
        train_unlabeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    return train_labeled_dataloader, train_unlabeled_dataloader, validation_dataloader