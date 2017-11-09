from hypylib.config import Config as ConfigBase
from hypylib.parameter import Parameter
import os


def path_join(*args):
    return os.path.join(os.path.expanduser("~"), *args)


class Config(ConfigBase):
    data_path = path_join("datasets", "aligned")
    log_base = path_join("siswslayer", "logs")
    model = Parameter(default="cnn", choices=['cnn', 'lws1', 'lws2', 'lcnn1', 'lcnn2'])
    optimizer = Parameter(default="adam", choices=['adam', 'rmsprop', 'nadam'])
    lr = 1e-4
    log = 'test'
    fold = 0
    log_dir = None
    batch_size = 32
    epochs = 150
    sample = 0.0
