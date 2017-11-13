import pandas as pd
import numpy as np
import h5py
from config import Config
import os.path as path


def preprocess():
    file = '/home/jos/datasets/fer2013/fer2013.csv'
    data = pd.read_csv(file)

    X_train, X_test = [], []
    y_train, y_test = [], []
    for i, row in data.iterrows():
        if row['Usage'] == 'Training':
            X_train.append(np.reshape([float(p) for p in row['pixels'].split()], (48, 48, 1)).astype(np.float32))
            y_train.append(row['emotion'])
        else:
            X_test.append(np.reshape([float(p) for p in row['pixels'].split()], (48, 48, 1)).astype(np.float32))
            y_test.append(row['emotion'])

    with h5py.File(path.join(Config.data_path, 'fer2013.h5'), mode='w') as f:
        f.create_dataset(name='train/images', data=X_train, dtype=np.float32)
        f.create_dataset(name='test/images', data=X_test, dtype=np.float32)
        f.create_dataset(name='train/labels', data=y_train, dtype=np.float32)
        f.create_dataset(name='test/labels', data=y_test, dtype=np.float32)


if __name__ == "__main__":
    Config.load()
    preprocess()
