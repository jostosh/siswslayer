import h5py
from numpy.core.umath import ceil
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.utils import to_categorical

from config import Config


class Dataset:

    def __init__(self, in_memory=True):
        self.h5f = h5py.File(Config.data_path + '/fold{}.hdf5'.format(Config.fold), 'r')

        if in_memory:
            # Loads the numpy arrays in memory
            self.X_train = self.h5f['train/images'][:]
            self.y_train = self.h5f['train/labels'][:]

            self.X_test = self.h5f['test/images'][:]
            self.y_test = self.h5f['test/labels'][:]

            self.ensure_one_hot()
        else:
            # Uses HDF5 access to files
            self.X_train = self.h5f['train/images']
            self.y_train = self.h5f['train/labels']

            self.X_test = self.h5f['test/images']
            self.y_test = self.h5f['test/labels']

        self.traingen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            rescale=1/255.,
            horizontal_flip=True
        )
        self.valgen = ImageDataGenerator(
            rescale=1/255.
        )

    def input_shape(self):
        return self.X_train.shape[1:]

    def n_classes(self):
        return self.y_train.shape[1]

    def ensure_one_hot(self):
        if self.y_test.ndim == 1:
            self.y_test = to_categorical(self.y_test)
        if self.y_train.ndim == 1:
            self.y_train = to_categorical(self.y_train)

    def generator(self, train=True):
        if train:
            return self.traingen.flow(
                self.X_train, self.y_train, batch_size=Config.batch_size
            )
        return self.valgen.flow(
            self.X_test, self.y_test, batch_size=Config.batch_size, shuffle=False
        )

    def validation_steps(self):
        return ceil(len(self.X_test) / Config.batch_size)

    def steps_per_epoch(self):
        return ceil(len(self.X_train) / Config.batch_size)

    def __del__(self):
        self.h5f.close()