from lws import LocalWeightSharing2D
from tensorflow.contrib.keras.python.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten,\
    LocallyConnected2D, ZeroPadding2D
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.losses import categorical_crossentropy
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop, Adam, Nadam
from config import Config
import tensorflow as tf
from initializers import get_initializer


opt = {
    'rmsprop': RMSprop,
    'adam': Adam,
    'nadam': Nadam
}


class CNN(Sequential):

    def __init__(self, input_shape=(227, 227, 3), n_classes=2, **kwargs):
        super().__init__(**kwargs)

        self.first_block(input_shape)
        self.second_block()
        self.third_block()
        self.final_block(n_classes)
        self.compile(
            loss=categorical_crossentropy, optimizer=opt[Config.optimizer](lr=Config.lr), metrics=['accuracy']
        )

    def first_block(self, input_shape):
        self.add(Conv2D(filters=96, kernel_size=7, strides=(4, 4), activation=tf.nn.elu, input_shape=input_shape))
        self.add(MaxPool2D(pool_size=(3, 3), strides=2))

    def second_block(self):
        self.add(Conv2D(filters=256, kernel_size=5, activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(3, 3), strides=2))

    def third_block(self):
        self.add(Conv2D(filters=384, kernel_size=3, activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(3, 3), strides=2))

    def final_block(self, n_classes):
        self.add(Flatten())
        self.add(Dense(512, activation=tf.nn.elu))
        self.add(Dropout(rate=0.5))
        self.add(Dense(512, activation=tf.nn.elu))
        self.add(Dropout(rate=0.5))

        self.add(Dense(n_classes, activation='softmax'))


class CNNSmall(CNN):

    def first_block(self, input_shape):
        self.add(Conv2D(filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, input_shape=input_shape,
                        padding='same'))
        self.add(Conv2D(filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))

    def second_block(self):
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same'))
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))

    def third_block(self):
        pass

    def final_block(self, n_classes):
        self.add(Flatten())
        self.add(Dense(256, activation=tf.nn.elu))
        self.add(Dropout(rate=0.5))
        self.add(Dense(256, activation=tf.nn.elu))
        self.add(Dropout(rate=0.5))

        self.add(Dense(n_classes, activation='softmax'))


class LWS1Small(CNNSmall):
    def second_block(self):
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same'))
        self.add(LocalWeightSharing2D(
            filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same',
            per_filter=Config.per_filter, gain=Config.gain, kernel_initializer=get_initializer(Config.init)
        ))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))


class LWS2Small(CNNSmall):
    def second_block(self):
        self.add(LocalWeightSharing2D(
            filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same',
            per_filter=Config.per_filter, gain=Config.gain, kernel_initializer=get_initializer(Config.init)
        ))
        self.add(LocalWeightSharing2D(
            filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same',
            per_filter=Config.per_filter, gain=Config.gain, kernel_initializer=get_initializer(Config.init)
        ))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))


class LCNN2Small(CNNSmall):
    def second_block(self):
        self.add(ZeroPadding2D())
        self.add(LocallyConnected2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(ZeroPadding2D())
        self.add(LocallyConnected2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))


class LCNN1Small(CNNSmall):
    def second_block(self):
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu, padding='same'))
        self.add(ZeroPadding2D())
        self.add(LocallyConnected2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))


class LWS1(CNN):

    def third_block(self):
        self.add(LocalWeightSharing2D(filters=384, kernel_size=3, activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(3, 3), strides=2))


class LWS2(LWS1):

    def second_block(self):
        self.add(LocalWeightSharing2D(filters=256, kernel_size=5, activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(3, 3), strides=2))


class LCNN1(CNN):

    def third_block(self):
        self.add(LocallyConnected2D(filters=384, kernel_size=3, activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(3, 3), strides=2))


class LCNN2(LCNN1):
    def second_block(self):
        self.add(LocallyConnected2D(filters=256, kernel_size=5, activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(3, 3), strides=2))


class CNNM(CNN):

    def first_block(self, input_shape):
        self.add(Conv2D(filters=32, kernel_size=5, strides=(2, 2), activation=tf.nn.elu, input_shape=input_shape))
        self.add(Conv2D(filters=32, kernel_size=5, strides=(2, 2), activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))

    def second_block(self):
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))

    def third_block(self):
        self.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))


class LWS1M(CNNM):
    def third_block(self):
        self.add(LocalWeightSharing2D(filters=256, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))


class LWS2M(LWS1M):
    def second_block(self):
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(LocalWeightSharing2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))


class LCNN1M(CNNM):
    def third_block(self):
        self.add(LocallyConnected2D(filters=256, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))


class LCNN2M(LCNN1M):
    def second_block(self):
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(LocallyConnected2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.elu))
        self.add(MaxPool2D(pool_size=(2, 2), strides=2))


model_dict = {
    'cnn': CNN,
    'lws1': LWS1,
    'lws2': LWS2,
    'lcnn1': LCNN1,
    'lcnn2': LCNN2,
    'lcnn1s': LCNN1Small,
    'lcnn2s': LCNN2Small,
    'lws1s': LWS1Small,
    'lw1s': LWS1Small,
    'lws2s': LWS2Small,
    'cnns': CNNSmall,
    'cnnm': CNNM,
    'lws1m': LWS1M,
    'lws2m': LWS2M,
    'lcnn1m': LCNN1M,
    'lcnn2m': LCNN2M
}