from hypylib.config import Config as ConfigBase
from hypylib.parameter import Parameter
import os


def path_join(*args):
    return os.path.join(os.path.expanduser("~"), *args)


class Config(ConfigBase):
    dataset = Parameter(default='adience', choices=['adience', 'lfw', 'mnist', 'svhn', 'cifar10', 'cifar100', 'iris'])
    data_path = path_join("datasets", "aligned")
    log_base = path_join("siswslayer", "logs")
    model = Parameter(
        default="cnn", choices=['cnn', 'lws1', 'lws2', 'lcnn1', 'lcnn2', 'lws1s', 'lws2s', 'lcnn1s', 'lcnn2s', 'cnns']
    )
    optimizer = Parameter(default="adam", choices=['adam', 'rmsprop', 'nadam'])
    lr = 1e-4
    log = 'test'
    fold = 0
    log_dir = None
    batch_size = 32
    epochs = 300
    verbose = 0
    exist_ok = False
    patience = 25
    workers = 4
    rotation_range = 10
    width_shift_range = 0.2
    height_shift_range = 0.2
    zoom_range = 0.2
    horizontal_flip = True
    in_memory = True
    width = 227
    height = 227
    resize = False
    crop = False
    cropx = Parameter(default=[61, 189], nargs=2, type=int)
    cropy = Parameter(default=[61, 189], nargs=2, type=int)
    kerosene_path = '/home/jos/datasets/kerosene'
    check_imports = False
    ensure_data = False