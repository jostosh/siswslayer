import h5py
from config import Config
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.callbacks import TerminateOnNaN, CSVLogger
from models import CNN, LWS
from numpy import ceil
from os.path import join
from os import makedirs
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class Dataset:

    def __init__(self):
        self.h5f = h5py.File(Config.data_path + '/fold{}.hdf5'.format(Config.fold), 'r')

        self.X_train = self.h5f['train/images'][()]
        self.y_train = self.h5f['train/labels'][()]

        self.X_test = self.h5f['test/images'][()]
        self.y_test = self.h5f['test/labels'][()]

        self.traingen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            rescale=1/255.
        )
        self.valgen = ImageDataGenerator(
            rescale=1/255.
        )

    def generator(self, train=True):
        if train:
            return self.traingen.flow(
                self.X_train, self.y_train, batch_size=Config.batch_size
            )
        return self.valgen.flow(
            self.X_test, self.y_test, batch_size=Config.batch_size, shuffle=False
        )

    def validation_steps(self):
        return ceil(len(self.X_test) // Config.batch_size)

    def steps_per_epoch(self):
        return ceil(len(self.X_train) // Config.batch_size)

    def __del__(self):
        self.h5f.close()


def train():
    dataset = Dataset()
    model = CNN() if Config.model == "cnn" else LWS()
    model.fit_generator(
        dataset.generator(), steps_per_epoch=dataset.steps_per_epoch(), epochs=Config.epochs,
        validation_data=dataset.generator(train=False), validation_steps=dataset.validation_steps(),
        verbose=1, workers=4, use_multiprocessing=True, callbacks=[
            CSVLogger(join(Config.log_dir, 'logs.csv')),
            TerminateOnNaN(),
        ]
    )

if __name__ == "__main__":
    Config.load()
    Config.log_dir = join(Config.log_base, Config.model, 'fold{}'.format(Config.fold))
    makedirs(Config.log_dir, exist_ok=True)
    train()
