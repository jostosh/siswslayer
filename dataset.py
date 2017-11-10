import h5py
from numpy.core.umath import ceil
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.utils import to_categorical
from config import Config
from scipy.misc import imresize
import abc
import numpy as np
from functools import partial


class DatasetBase(abc.ABC):

    def __init__(self):
        self.h5f, self.X_train, self.y_train, self.X_test, self.y_test = self.load_data()

        # Perform some preprocessing steps
        self.preprocess()

        # Data augmentation
        self.train_generator = self.train_generator()

        # Validation generator
        self.validation_generator = ImageDataGenerator(
            rescale=self.determine_rescale()
        )

    def preprocess(self):
        self.ensure_one_hot()
        self.ensure_channels_last()
        # self.crop()
        self.ensure_resized()

    def train_generator(self):
        return ImageDataGenerator(
            rotation_range=Config.rotation_range,
            width_shift_range=Config.width_shift_range,
            height_shift_range=Config.height_shift_range,
            zoom_range=Config.zoom_range,
            rescale=self.determine_rescale(),
            horizontal_flip=Config.horizontal_flip
        )

    def determine_rescale(self):
        if self.X_train[0].max() > 1.0:
            return 1/255.0

    def ensure_channels_last(self):
        if self.X_train.shape[1] in [1, 3, 4] and self.X_train.shape[-1] not in [1, 3, 4]:
            self.X_train = self.X_train.transpose((0, 2, 3, 1))
            self.X_test = self.X_test.transpose((0, 2, 3, 1))

    def ensure_resized(self):
        if Config.resize:
            self.X_test = np.asarray(map(lambda im: imresize(im, Config.widhth, Config.height), self.X_test))
            self.X_train = np.asarray(map(lambda im: imresize(im, (Config.width, Config.height)), self.X_train))

    def crop(self):
        self.X_train = np.asarray(map(
            lambda im: im[Config.cropy[0]:Config.cropy[1], Config.cropx[0]:Config.cropx[1]],
            self.X_train
        ))
        self.X_test = np.asarray(map(
            lambda im: im[Config.cropy[0]:Config.cropy[1], Config.cropx[0]:Config.cropx[1]],
            self.X_test
        ))

    @abc.abstractmethod
    def load_data(self) -> (h5py.File, h5py.Dataset, h5py.Dataset, h5py.Dataset, h5py.Dataset, h5py.Dataset):
        """ Loads the data """

    def input_shape(self):
        return self.X_train.shape[1:]

    def n_classes(self):
        return self.y_train.shape[1]

    def ensure_one_hot(self):
        if self.y_test.ndim == 1 or (self.y_test.ndim == 2 and self.y_test.shape[1] == 1):
            self.y_test = to_categorical(self.y_test)
        if self.y_train.ndim == 1 or (self.y_train.ndim == 2 and self.y_train.shape[1] == 1):
            self.y_train = to_categorical(self.y_train)

    def generator(self, train=True):
        if train:
            return self.train_generator.flow(
                self.X_train, self.y_train, batch_size=Config.batch_size
            )
        return self.validation_generator.flow(
            self.X_test, self.y_test, batch_size=Config.batch_size, shuffle=False
        )

    def validation_steps(self):
        return ceil(len(self.X_test) / Config.batch_size)

    def steps_per_epoch(self):
        return ceil(len(self.X_train) / Config.batch_size)

    @staticmethod
    def load_kerosene(loader):
        return loader(datadir=Config.kerosene_path)

    def __del__(self):
        if self.h5f:
            self.h5f.close()


class Adience(DatasetBase):

    def load_data(self):

        self.h5f = h5py.File(Config.data_path + '/fold{}.hdf5'.format(Config.fold), 'r')

        if Config.in_memory:
            # Loads the numpy arrays in memory
            self.X_train = self.h5f['train/images'][:]
            self.y_train = self.h5f['train/labels'][:]

            self.X_test = self.h5f['test/images'][:]
            self.y_test = self.h5f['test/labels'][:]

        else:
            # Uses HDF5 access to files
            self.X_train = self.h5f['train/images']
            self.y_train = self.h5f['train/labels']

            self.X_test = self.h5f['test/images']
            self.y_test = self.h5f['test/labels']

        return self.h5f, self.X_train, self.y_train, self.X_test, self.y_test


class MNIST(DatasetBase):

    def load_data(self):
        from kerosene.datasets.mnist import load_data
        (X_train, y_train), (X_test, y_test) = DatasetBase.load_kerosene(load_data)
        return None, X_train, y_train, X_test, y_test

    def train_generator(self):
        return ImageDataGenerator(
            rotation_range=Config.rotation_range,
            width_shift_range=Config.width_shift_range,
            height_shift_range=Config.height_shift_range,
            zoom_range=Config.zoom_range,
            rescale=self.determine_rescale()
        )


class SVHN(DatasetBase):

    def load_data(self):
        from kerosene.datasets.svhn2 import load_data
        (X_train, y_train), (X_test, y_test) = DatasetBase.load_kerosene(load_data)
        return None, X_train, y_train, X_test, y_test

    def train_generator(self):
        return ImageDataGenerator(
            rotation_range=Config.rotation_range,
            width_shift_range=Config.width_shift_range,
            height_shift_range=Config.height_shift_range,
            zoom_range=Config.zoom_range,
            rescale=self.determine_rescale()
        )


class LFW(DatasetBase):

    def preprocess(self):
        self.ensure_one_hot()
        self.ensure_channels_last()
        self.crop()
        self.ensure_resized()

    def load_data(self):
        from lfw_fuel.lfw import load_data
        (X_train, y_train), (X_test, y_test) = DatasetBase.load_kerosene(partial(load_data, format='deepfunneled'))
        return None, X_train, y_train, X_test, y_test


class Cifar10(DatasetBase):

    def load_data(self):
        from kerosene.datasets.cifar10 import load_data
        (X_train, y_train), (X_test, y_test) = DatasetBase.load_kerosene(load_data)
        return None, X_train, y_train, X_test, y_test


class Cifar100(DatasetBase):

    def load_data(self):
        from kerosene.datasets.cifar100 import load_data
        (X_train, y_train), (X_test, y_test) = DatasetBase.load_kerosene(load_data)
        return None, X_train, y_train, X_test, y_test



def get_dataset(name):
    return {
        'adience': Adience,
        'mnist': MNIST,
        'svhn': SVHN,
        'lfw': LFW,
        'cifar10': Cifar10
    }[name]()
