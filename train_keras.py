import os
from os import makedirs
from os.path import join

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.callbacks import TerminateOnNaN, CSVLogger, LearningRateScheduler
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
    print("Loaded data")
    model = model_dict[Config.model](input_shape=dataset.input_shape(), n_classes=dataset.n_classes())
    print("Built model")
    model.fit_generator(
        dataset.generator(), steps_per_epoch=dataset.steps_per_epoch(), epochs=Config.epochs,
        validation_data=dataset.generator(train=False), validation_steps=dataset.validation_steps(),
        verbose=Config.verbose, workers=Config.workers, use_multiprocessing=True, callbacks=[
            CSVLogger(join(Config.log_dir, 'logs.csv')),
            TerminateOnNaN(),
            LearningRateScheduler(lambda t: Config.lr * (Config.lr_decay ** t)),
            EstimateTimeRemaining(total_epochs=Config.epochs)
        ]
    )
    if Config.store:
        model.save(join(Config.log_dir, 'model.h5'))

if __name__ == "__main__":
    Config.load()
    if Config.check_imports:
        print("Imports succeeded!")
        exit(0)
    Config.log_dir = join(Config.log_base, Config.dataset, Config.model, Config.log_infix, 'fold{}'.format(Config.fold))
    makedirs(Config.log_dir, exist_ok=Config.exist_ok)
    print("Logging at {}".format(Config.log_dir))
    train()
