from config import ConfigBase
import pandas as pd
import numpy as np
import os
import h5py
from scipy import misc
import matplotlib.pylab as plt
from preprocessing.frontalizer import Frontalizer


class AdienceConfig(ConfigBase):
    data_path = '/home/jos/datasets/aligned'
    width = 227
    height = 227
    folds = 5
    frontal = False
    show = False


def load_fold(idx):
    df = pd.read_csv('/home/jos/datasets/aligned/fold_frontal_{}_data.txt'.format(idx), sep='\t').dropna()
    df = df[df['gender'] != 'u']

    images = []
    labels = []

    for i, row in df.iterrows():
        if i % 1000 == 0:
            print("Image index:", i)

        path = os.path.join(AdienceConfig.data_path, row['user_id'], 'landmark_aligned_face.{}.{}'.format(
            row['face_id'], row['original_image']
        ))
        gender = row['gender']
        if not os.path.exists(path):
            continue

        img = misc.imread(path)
        if frontalizer:
            _, img = frontalizer.transform(img)
        if AdienceConfig.show:
            plt.imshow(img)
            plt.show()
        images.append(misc.imresize(misc.imread(path), (AdienceConfig.width, AdienceConfig.height)))
        labels.append(np.asarray([1, 0], dtype='float') if gender == 'm' else np.asarray([0, 1], dtype='float'))

    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels



def preprocess():
    all_images = []
    all_labels = []
    for i in range(AdienceConfig.folds):
        images, labels = load_fold(i)
        all_images.append(images)
        all_labels.append(labels)

    for i in range(AdienceConfig.folds):
        with h5py.File(AdienceConfig.data_path + "/fold{}.hdf5".format(i), "w") as f:
            train_sets = list(range(AdienceConfig.folds))
            train_sets.remove(i)

            f.create_dataset("train/images", data=np.concatenate([all_images[j] for j in train_sets]))
            f.create_dataset("train/labels", data=np.concatenate([all_labels[j] for j in train_sets]))

            f.create_dataset("test/images", data=all_images[i])
            f.create_dataset("test/labels", data=all_labels[i])


if __name__ == "__main__":
    AdienceConfig.load()
    frontalizer = Frontalizer() if AdienceConfig.frontal else None
    preprocess()