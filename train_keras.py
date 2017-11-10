import os
from os import makedirs
from os.path import join

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.callbacks import TerminateOnNaN, CSVLogger, ReduceLROnPlateau
from callbacks import EstimateTimeRemaining

from config import Config
from dataset import get_dataset
from models import model_dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def train():
    dataset = get_dataset(Config.dataset)
    if Config.ensure_data:
        print("Dataset {} ready".format(Config.dataset))
        return
    model = model_dict[Config.model](input_shape=dataset.input_shape(), n_classes=dataset.n_classes())
    model.fit_generator(
        dataset.generator(), steps_per_epoch=dataset.steps_per_epoch(), epochs=Config.epochs,
        validation_data=dataset.generator(train=False), validation_steps=dataset.validation_steps(),
        verbose=Config.verbose, workers=Config.workers, use_multiprocessing=True, callbacks=[
            CSVLogger(join(Config.log_dir, 'logs.csv')),
            TerminateOnNaN(),
            ReduceLROnPlateau(patience=Config.patience),
            EstimateTimeRemaining(total_epochs=Config.epochs)
        ]
    )

if __name__ == "__main__":
    Config.load()
    if Config.check_imports:
        print("Imports succeeded!")
        exit(0)
    Config.log_dir = join(Config.log_base, Config.dataset, Config.model, 'fold{}'.format(Config.fold))
    makedirs(Config.log_dir, exist_ok=Config.exist_ok)
    train()
