import os
from os import makedirs
from os.path import join

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.callbacks import TerminateOnNaN, CSVLogger, ReduceLROnPlateau
from callbacks import EstimateTimeRemaining

from config import Config
from dataset import Dataset
from models import model_dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def train():
    dataset = Dataset()
    model = model_dict[Config.model](input_shape=dataset.input_shape(), n_classes=dataset.n_classes())
    model.fit_generator(
        dataset.generator(), steps_per_epoch=dataset.steps_per_epoch(), epochs=Config.epochs,
        validation_data=dataset.generator(train=False), validation_steps=dataset.validation_steps(),
        verbose=1, workers=4, use_multiprocessing=True, callbacks=[
            CSVLogger(join(Config.log_dir, 'logs.csv')),
            TerminateOnNaN(),
            ReduceLROnPlateau(patience=25),
            EstimateTimeRemaining(total_epochs=Config.epochs)
        ]
    )

if __name__ == "__main__":
    Config.load()
    Config.log_dir = join(Config.log_base, Config.model, 'fold{}'.format(Config.fold))
    makedirs(Config.log_dir)
    train()
